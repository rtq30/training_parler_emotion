import torch
from torchaudio.pipelines import SQUIM_OBJECTIVE
import torchaudio
import evaluate
# Hardy: Updated imports - removed HuggingFace SER imports, added funasr and file handling
from transformers import (
    AutoModel,
    AutoProcessor,
    pipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperTokenizerFast,
)
from accelerate.utils.memory import release_memory
import numpy as np
# New imports for emotion2vec
from funasr import AutoModel as FunASRAutoModel
import soundfile as sf
import tempfile
import os


def clap_similarity(clap_model_name_or_path, texts, audios, device, input_sampling_rate=44100):
    clap = AutoModel.from_pretrained(clap_model_name_or_path)
    clap_processor = AutoProcessor.from_pretrained(clap_model_name_or_path)
    output_sampling_rate = clap_processor.feature_extractor.sampling_rate
    if input_sampling_rate != output_sampling_rate:
        audios = [
            torchaudio.functional.resample(torch.from_numpy(audio), input_sampling_rate, output_sampling_rate).numpy()
            for audio in audios
        ]
    clap_inputs = clap_processor(
        text=texts, audios=audios, padding=True, return_tensors="pt", sampling_rate=output_sampling_rate
    ).to(device)

    clap.to(device)
    with torch.no_grad():
        text_features = clap.get_text_features(
            clap_inputs["input_ids"], attention_mask=clap_inputs.get("attention_mask", None)
        )
        audio_features = clap.get_audio_features(clap_inputs["input_features"])

        cosine_sim = torch.nn.functional.cosine_similarity(audio_features, text_features, dim=1, eps=1e-8).mean()

    cosine_sim = cosine_sim.to("cpu")

    clap.to("cpu")
    clap, clap_inputs, audio_features, text_features = release_memory(clap, clap_inputs, audio_features, text_features)
    return cosine_sim


def si_sdr(audios, device, input_sampling_rate=44100):
    max_audio_length = 15 * SQUIM_OBJECTIVE.sample_rate
    model = SQUIM_OBJECTIVE.get_model().to((device))

    output_sampling_rate = SQUIM_OBJECTIVE.sample_rate
    if input_sampling_rate != output_sampling_rate:
        audios = [
            torchaudio.functional.resample(
                torch.tensor(audio)[None, :].to(device).float(), input_sampling_rate, output_sampling_rate
            )
            for audio in audios
        ]

    def apply_squim(waveform):
        with torch.no_grad():
            waveform = waveform[:, : min(max_audio_length, waveform.shape[1])]
            _, _, sdr_sample = model(waveform)
            sdr_sample = sdr_sample.cpu()[0]
        return sdr_sample

    si_sdrs = [apply_squim(audio) for audio in audios]
    audios, model = release_memory(audios, model)
    return si_sdrs


def wer(
        asr_model_name_or_path,
        prompts,
        audios,
        device,
        per_device_eval_batch_size,
        sampling_rate,
        noise_level_to_compute_clean_wer,
        si_sdr_measures,
):
    metric = evaluate.load("wer")
    asr_pipeline = pipeline(
        task="automatic-speech-recognition",
        model=asr_model_name_or_path, device=device, chunk_length_s=25.0)
    # Hardy: Modified here by adding task..., on Jun 27th
    return_language = None
    if isinstance(asr_pipeline.model, WhisperForConditionalGeneration):
        return_language = True

    transcriptions = asr_pipeline(
        [{"raw": audio, "sampling_rate": sampling_rate} for audio in audios],
        batch_size=int(per_device_eval_batch_size),
        return_language=return_language,
    )

    if isinstance(asr_pipeline.tokenizer, (WhisperTokenizer, WhisperTokenizerFast)):
        tokenizer = asr_pipeline.tokenizer
    else:
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")

    english_normalizer = tokenizer.normalize
    basic_normalizer = tokenizer.basic_normalize

    normalized_predictions = []
    normalized_references = []

    for pred, ref in zip(transcriptions, prompts):
        normalizer = (
            english_normalizer
            if isinstance(pred.get("chunks", None), list) and pred["chunks"][0].get("language", None) == "english"
            else basic_normalizer
        )
        norm_ref = normalizer(ref)
        if len(norm_ref) > 0:
            norm_pred = normalizer(pred["text"])
            normalized_predictions.append(norm_pred)
            normalized_references.append(norm_ref)

    word_error = 100
    clean_word_error = None
    noisy_word_error = None
    percent_clean_samples = 0
    if len(normalized_references) > 0:
        word_error = 100 * metric.compute(predictions=normalized_predictions, references=normalized_references)

        if noise_level_to_compute_clean_wer and si_sdr_measures:
            si_sdr_measures = np.array(si_sdr_measures)
            mask = si_sdr_measures >= noise_level_to_compute_clean_wer
            if mask.any():
                clean_word_error = 100 * metric.compute(
                    predictions=np.array(normalized_predictions)[mask], references=np.array(normalized_references)[mask]
                )
                if not mask.all():
                    noisy_word_error = 100 * metric.compute(
                        predictions=np.array(normalized_predictions)[~mask],
                        references=np.array(normalized_references)[~mask]
                    )
                else:
                    noisy_word_error = 0
                percent_clean_samples = mask.sum() / len(mask)

    asr_pipeline.model.to("cpu")
    asr_pipeline = release_memory(asr_pipeline)
    return word_error, [t["text"] for t in transcriptions], clean_word_error, noisy_word_error, percent_clean_samples


# Hardy: The new emotion2vec-based SER function
def speech_emotion_recognition(
        ser_model_name_or_path,
        audios,
        emotion_labels,
        device,
        per_device_eval_batch_size,
        input_sampling_rate=44100,
        max_duration=30.0,
):
    """
    Compute emotion recognition accuracy using emotion2vec model with detailed logging.
    Modified to use FunASR AutoModel and handle multiple acceptable labels per emotion.
    """
    # Emotion mapping from dataset labels to emotion2vec labels
    # Some emotions can map to multiple acceptable labels
    EMOTION_MAPPING = {
        "Happy": ["开心/happy"],
        "Confused": ["中立/neutral", "其他/other", "吃惊/surprised"],
        "Neutral": ["中立/neutral"],
        "Laughing": ["开心/happy"],
        "Sad": ["难过/sad"],
        "Whisper": ["中立/neutral", "其他/other", "<unk>"],
        "Emphasis": ["中立/neutral", "其他/other", "<unk>", "吃惊/surprised"],
    }

    # Track which emotions don't have direct mappings
    EMOTIONS_WITH_MULTIPLE_MAPPINGS = {"Confused", "Whisper", "Emphasis"}

    # Load emotion2vec model using FunASR
    print(f"\nLoading emotion2vec model from: {ser_model_name_or_path}")
    model = FunASRAutoModel(
        model=ser_model_name_or_path,
        hub="ms",  # modelscope hub
        use_cache=True
    )

    # emotion2vec model labels in order
    model_labels = ['生气/angry', '厌恶/disgusted', '恐惧/fearful', '开心/happy',
                    '中立/neutral', '其他/other', '难过/sad', '吃惊/surprised', '<unk>']

    # Log model information
    print("\n" + "=" * 60)
    print("SER MODEL INFORMATION (emotion2vec)")
    print("=" * 60)
    print(f"Model: {ser_model_name_or_path}")
    print(f"Available emotions in model: {model_labels}")
    print(f"Number of evaluation samples: {len(emotion_labels)}")
    print(f"Unique emotions in dataset: {set(emotion_labels)}")
    print(f"Emotion mapping being used:")
    for key, values in EMOTION_MAPPING.items():
        print(f"  {key}: {values}")
    print("=" * 60 + "\n")

    predictions = []
    probabilities = []

    # Create a temporary directory for audio files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Process audios in batches
        for i in range(0, len(audios), per_device_eval_batch_size):
            batch_audios = audios[i:i + per_device_eval_batch_size]
            batch_predictions = []
            batch_probabilities = []

            for j, audio in enumerate(batch_audios):
                # Convert numpy array to temporary WAV file
                temp_audio_path = os.path.join(temp_dir, f"temp_audio_{i}_{j}.wav")

                # Ensure audio is in the correct format
                if isinstance(audio, torch.Tensor):
                    audio = audio.cpu().numpy()

                # Save audio to temporary file
                sf.write(temp_audio_path, audio, input_sampling_rate)

                try:
                    # Run emotion2vec inference
                    res = model.generate(
                        temp_audio_path,
                        output_dir=temp_dir,
                        granularity="utterance",
                        extract_embedding=False
                    )

                    # Extract results
                    labels = res[0]['labels']
                    scores = res[0]['scores']

                    # Get the model's top prediction
                    max_idx = np.argmax(scores)
                    predicted_label = labels[max_idx]

                    # Create probability vector in the order of model_labels
                    label_scores = {label: score for label, score in zip(labels, scores)}
                    prob_vector = np.array([label_scores.get(label, 0.0) for label in model_labels])

                    batch_predictions.append(predicted_label)
                    batch_probabilities.append(prob_vector)

                except Exception as e:
                    print(f"Error processing audio {i}_{j}: {e}")
                    # Add a dummy prediction in case of error
                    batch_predictions.append("<unk>")
                    batch_probabilities.append(np.zeros(len(model_labels)))

                # Clean up temporary file
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)

            predictions.extend(batch_predictions)
            probabilities.extend(batch_probabilities)

    # Release model memory
    model = release_memory(model)

    # Detailed logging of all predictions
    print("\n" + "=" * 60)
    print("DETAILED PREDICTION RESULTS")
    print("=" * 60)
    print(f"{'Sample':>6} | {'True Label':>10} | {'Acceptable':>30} | {'Predicted':>15} | {'Match':>5} | Confidence")
    print("-" * 90)

    # Compute metrics
    correct = 0
    per_emotion_correct = {}
    per_emotion_total = {}
    emotion_specific_scores = []
    emotions_with_multiple_results = {}
    detailed_results = []

    for idx, (pred, true_label, prob_dist) in enumerate(zip(predictions, emotion_labels, probabilities)):
        # Track per-emotion totals
        if true_label not in per_emotion_total:
            per_emotion_total[true_label] = 0
            per_emotion_correct[true_label] = 0
        per_emotion_total[true_label] += 1

        # Get acceptable emotions for this true label
        acceptable_labels = EMOTION_MAPPING.get(true_label, ["中立/neutral"])  # Default to neutral if not found

        # Check if prediction matches any acceptable emotion
        match = pred in acceptable_labels
        if match:
            correct += 1
            per_emotion_correct[true_label] += 1

        # Get confidence for the predicted label
        pred_idx = model_labels.index(pred) if pred in model_labels else -1
        confidence = prob_dist[pred_idx] if pred_idx >= 0 else 0.0

        # Print detailed result for first 50 samples
        if idx < 50:
            acceptable_str = ", ".join(acceptable_labels[:2]) + ("..." if len(acceptable_labels) > 2 else "")
            match_symbol = "✓" if match else "✗"
            print(
                f"{idx:>6} | {true_label:>10} | {acceptable_str:>30} | {pred:>15} | {match_symbol:>5} | {confidence:.2%}")

        # Store detailed result
        detailed_results.append({
            "sample_idx": idx,
            "true_label": true_label,
            "acceptable_predictions": acceptable_labels,
            "actual_prediction": pred,
            "match": match,
            "confidence": float(confidence),
            "all_probabilities": {model_labels[i]: float(prob_dist[i]) for i in range(len(prob_dist))}
        })

        # For emotions with multiple mappings, track the distribution
        if true_label in EMOTIONS_WITH_MULTIPLE_MAPPINGS:
            if true_label not in emotions_with_multiple_results:
                emotions_with_multiple_results[true_label] = {
                    "predictions": [],
                    "probabilities": []
                }
            emotions_with_multiple_results[true_label]["predictions"].append(pred)
            emotions_with_multiple_results[true_label]["probabilities"].append(prob_dist)

        # Get average probability score for all acceptable emotions
        acceptable_scores = []
        for acceptable_label in acceptable_labels:
            if acceptable_label in model_labels:
                label_idx = model_labels.index(acceptable_label)
                acceptable_scores.append(prob_dist[label_idx])
        emotion_score = np.mean(acceptable_scores) if acceptable_scores else 0.0
        emotion_specific_scores.append(emotion_score)

    # Calculate overall accuracy
    accuracy = correct / len(emotion_labels) if len(emotion_labels) > 0 else 0.0

    # Calculate per-emotion accuracy
    per_emotion_accuracy = {}
    for emotion in per_emotion_total:
        if per_emotion_total[emotion] > 0:
            per_emotion_accuracy[emotion] = per_emotion_correct[emotion] / per_emotion_total[emotion]
        else:
            per_emotion_accuracy[emotion] = 0.0

    # Calculate average emotion-specific score
    avg_emotion_score = np.mean(emotion_specific_scores) if emotion_specific_scores else 0.0

    # Prepare detailed results for emotions with multiple mappings
    unmapped_stats = {}
    for emotion, results in emotions_with_multiple_results.items():
        pred_counts = {}
        for pred in results["predictions"]:
            pred_counts[pred] = pred_counts.get(pred, 0) + 1

        avg_probs = np.mean(results["probabilities"], axis=0)
        top_predictions = []
        for i, prob in enumerate(avg_probs):
            if prob > 0.1:
                top_predictions.append({
                    "emotion": model_labels[i],
                    "avg_probability": float(prob)
                })

        unmapped_stats[emotion] = {
            "acceptable_labels": EMOTION_MAPPING[emotion],
            "prediction_distribution": pred_counts,
            "top_predictions": sorted(top_predictions, key=lambda x: x["avg_probability"], reverse=True)
        }

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Overall Accuracy: {accuracy:.2%} ({correct}/{len(emotion_labels)})")
    print(f"Average Confidence Score: {avg_emotion_score:.2%}")
    print("\nPer-Emotion Accuracy:")
    for emotion in sorted(per_emotion_total.keys()):
        acc = per_emotion_accuracy[emotion]
        correct_count = per_emotion_correct[emotion]
        total = per_emotion_total[emotion]
        print(f"  {emotion:>10}: {acc:>6.2%} ({correct_count}/{total})")

    # Show distribution for emotions with multiple mappings
    if emotions_with_multiple_results:
        print("\n" + "=" * 60)
        print("PREDICTION DISTRIBUTION FOR MULTI-MAPPING EMOTIONS")
        print("=" * 60)
        for emotion, stats in unmapped_stats.items():
            print(f"\n{emotion} (acceptable: {', '.join(stats['acceptable_labels'])}):")
            for pred_emotion, count in sorted(stats['prediction_distribution'].items(),
                                              key=lambda x: x[1], reverse=True):
                percentage = count / per_emotion_total[emotion] * 100
                print(f"  → {pred_emotion}: {count} ({percentage:.1f}%)")

    print("=" * 60 + "\n")

    return {
        "ser_accuracy": accuracy * 100,
        "ser_avg_emotion_score": avg_emotion_score * 100,
        "ser_per_emotion_accuracy": per_emotion_accuracy,
        "ser_predictions": predictions,
        "ser_probabilities": probabilities,
        "ser_emotion_mapping": EMOTION_MAPPING,
        "ser_unmapped_emotion_stats": unmapped_stats,
        "ser_detailed_results": detailed_results,
    }
