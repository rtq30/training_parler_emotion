import torch
from torchaudio.pipelines import SQUIM_OBJECTIVE
import torchaudio
import evaluate
# Hardy: I added AutoModelForAudioClassification and AutoFeatureExtractor here to import the SER model
from transformers import (
    AutoModel,
    AutoProcessor,
    pipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperTokenizerFast,
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
)
from accelerate.utils.memory import release_memory
import numpy as np


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
                        predictions=np.array(normalized_predictions)[~mask], references=np.array(normalized_references)[~mask]
                    )
                else:
                    noisy_word_error = 0
                percent_clean_samples = mask.sum() / len(mask)

    asr_pipeline.model.to("cpu")
    asr_pipeline = release_memory(asr_pipeline)
    return word_error, [t["text"] for t in transcriptions], clean_word_error, noisy_word_error, percent_clean_samples
    
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
    Compute emotion recognition accuracy using a pre-trained SER model with detailed logging.
    """
    # Emotion mapping from your labels to SER model labels
    EMOTION_MAPPING = {
        "Happy": "happy",
        "Confused": "surprised",
        "Neutral": "neutral",
        "Laughing": "happy",
        "Sad": "sad",
        "Whisper": "neutral",
        "Emphasis": "neutral",
    }

    # Track which emotions don't have direct mappings
    UNMAPPED_EMOTIONS = {"Confused", "Laughing", "Whisper", "Emphasis"}

    # Load model and feature extractor
    model = AutoModelForAudioClassification.from_pretrained(ser_model_name_or_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(ser_model_name_or_path, do_normalize=True)
    model.to(device)
    model.eval()

    output_sampling_rate = feature_extractor.sampling_rate
    max_length = int(output_sampling_rate * max_duration)
    id2label = model.config.id2label
    label2id = {v: k for k, v in id2label.items()}

    # Log model information
    print("\n" + "=" * 60)
    print("SER MODEL INFORMATION")
    print("=" * 60)
    print(f"Model: {ser_model_name_or_path}")
    print(f"Available emotions in model: {list(label2id.keys())}")
    print(f"Number of evaluation samples: {len(emotion_labels)}")
    print(f"Unique emotions in dataset: {set(emotion_labels)}")
    print(f"Emotion mapping being used: {EMOTION_MAPPING}")
    print("=" * 60 + "\n")

    predictions = []
    probabilities = []

    # Process audios in batches
    for i in range(0, len(audios), per_device_eval_batch_size):
        batch_audios = audios[i:i + per_device_eval_batch_size]

        # Resample if necessary
        if input_sampling_rate != output_sampling_rate:
            batch_audios = [
                torchaudio.functional.resample(
                    torch.from_numpy(audio), input_sampling_rate, output_sampling_rate
                ).numpy()
                for audio in batch_audios
            ]

        # Prepare batch inputs
        batch_inputs = []
        for audio in batch_audios:
            if len(audio) > max_length:
                audio = audio[:max_length]
            else:
                audio = np.pad(audio, (0, max_length - len(audio)))
            batch_inputs.append(audio)

        # Extract features
        inputs = feature_extractor(
            batch_inputs,
            sampling_rate=output_sampling_rate,
            return_tensors="pt",
            padding=True,
        ).to(device)

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            predicted_ids = torch.argmax(logits, dim=-1)
            batch_predictions = [id2label[pred_id.item()] for pred_id in predicted_ids]
            batch_probabilities = probs.cpu().numpy()

            predictions.extend(batch_predictions)
            probabilities.extend(batch_probabilities)

    # Release model memory
    model.to("cpu")
    model = release_memory(model)

    # Detailed logging of all predictions
    print("\n" + "=" * 60)
    print("DETAILED PREDICTION RESULTS")
    print("=" * 60)
    print(
        f"{'Sample':>6} | {'True Label':>10} | {'Expected':>10} | {'Predicted':>10} | {'Match':>5} | Top 3 Predictions")
    print("-" * 100)

    # Compute metrics
    correct = 0
    mapped_correct = 0
    per_emotion_correct = {}
    per_emotion_total = {}
    emotion_specific_scores = []
    unmapped_emotion_results = {}

    # Create a detailed results list for saving/analysis
    detailed_results = []

    for idx, (pred, true_label, prob_dist) in enumerate(zip(predictions, emotion_labels, probabilities)):
        # Track per-emotion totals
        if true_label not in per_emotion_total:
            per_emotion_total[true_label] = 0
            per_emotion_correct[true_label] = 0
        per_emotion_total[true_label] += 1

        # Get mapped emotion for evaluation
        mapped_emotion = EMOTION_MAPPING.get(true_label, "neutral")

        # Check if prediction matches mapped emotion
        match = pred == mapped_emotion
        if match:
            mapped_correct += 1
            per_emotion_correct[true_label] += 1

        # Get top 3 predictions with probabilities
        top_3_indices = np.argsort(prob_dist)[-3:][::-1]
        top_3_preds = [(id2label[i], prob_dist[i]) for i in top_3_indices]
        top_3_str = " | ".join([f"{label}:{prob:.2%}" for label, prob in top_3_preds])

        # Print detailed result for this sample
        match_symbol = "✓" if match else "✗"
        print(f"{idx:>6} | {true_label:>10} | {mapped_emotion:>10} | {pred:>10} | {match_symbol:>5} | {top_3_str}")

        # Store detailed result
        detailed_results.append({
            "sample_idx": idx,
            "true_label": true_label,
            "expected_prediction": mapped_emotion,
            "actual_prediction": pred,
            "match": match,
            "confidence": float(np.max(prob_dist)),
            "top_3_predictions": {label: float(prob) for label, prob in top_3_preds},
            "all_probabilities": {id2label[i]: float(prob_dist[i]) for i in range(len(prob_dist))}
        })

        # For unmapped emotions, track the distribution
        if true_label in UNMAPPED_EMOTIONS:
            if true_label not in unmapped_emotion_results:
                unmapped_emotion_results[true_label] = {
                    "predictions": [],
                    "probabilities": []
                }
            unmapped_emotion_results[true_label]["predictions"].append(pred)
            unmapped_emotion_results[true_label]["probabilities"].append(prob_dist)

        # Get probability score for the mapped emotion
        if mapped_emotion in label2id:
            mapped_emotion_id = label2id[mapped_emotion]
            emotion_score = prob_dist[mapped_emotion_id]
            emotion_specific_scores.append(emotion_score)
        else:
            emotion_specific_scores.append(0.0)

    # Calculate overall accuracy
    accuracy = mapped_correct / len(emotion_labels) if len(emotion_labels) > 0 else 0.0

    # Calculate per-emotion accuracy
    per_emotion_accuracy = {}
    for emotion in per_emotion_total:
        if per_emotion_total[emotion] > 0:
            per_emotion_accuracy[emotion] = per_emotion_correct[emotion] / per_emotion_total[emotion]
        else:
            per_emotion_accuracy[emotion] = 0.0

    # Calculate average emotion-specific score
    avg_emotion_score = np.mean(emotion_specific_scores) if emotion_specific_scores else 0.0

    # Prepare detailed results for unmapped emotions
    unmapped_stats = {}
    for emotion, results in unmapped_emotion_results.items():
        pred_counts = {}
        for pred in results["predictions"]:
            pred_counts[pred] = pred_counts.get(pred, 0) + 1

        avg_probs = np.mean(results["probabilities"], axis=0)
        top_predictions = []
        for i, prob in enumerate(avg_probs):
            if prob > 0.1:
                top_predictions.append({
                    "emotion": id2label[i],
                    "avg_probability": float(prob)
                })

        unmapped_stats[emotion] = {
            "mapped_to": EMOTION_MAPPING[emotion],
            "prediction_distribution": pred_counts,
            "top_predictions": sorted(top_predictions, key=lambda x: x["avg_probability"], reverse=True)
        }

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Overall Accuracy: {accuracy:.2%} ({mapped_correct}/{len(emotion_labels)})")
    print(f"Average Confidence Score: {avg_emotion_score:.2%}")
    print("\nPer-Emotion Accuracy:")
    for emotion in sorted(per_emotion_total.keys()):
        acc = per_emotion_accuracy[emotion]
        correct = per_emotion_correct[emotion]
        total = per_emotion_total[emotion]
        print(f"  {emotion:>10}: {acc:>6.2%} ({correct}/{total})")

    # Confusion matrix style summary
    print("\n" + "=" * 60)
    print("PREDICTION DISTRIBUTION PER TRUE EMOTION")
    print("=" * 60)
    for true_emotion in sorted(per_emotion_total.keys()):
        # Count predictions for this true emotion
        pred_distribution = {}
        for idx, (true_label, pred) in enumerate(zip(emotion_labels, predictions)):
            if true_label == true_emotion:
                pred_distribution[pred] = pred_distribution.get(pred, 0) + 1

        print(f"\n{true_emotion}:")
        for pred_emotion, count in sorted(pred_distribution.items(), key=lambda x: x[1], reverse=True):
            percentage = count / per_emotion_total[true_emotion] * 100
            print(f"  → {pred_emotion}: {count} ({percentage:.1f}%)")

    print("=" * 60 + "\n")

    # Save detailed results to a JSON file for further analysis
    import json
    import os
    import time
    results_dir = "ser_evaluation_results"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"ser_detailed_results_{timestamp}.json")

    with open(results_file, "w") as f:
        json.dump({
            "summary": {
                "total_samples": len(emotion_labels),
                "overall_accuracy": accuracy * 100,
                "avg_emotion_score": avg_emotion_score * 100,
                "per_emotion_accuracy": {k: v * 100 for k, v in per_emotion_accuracy.items()},
                "per_emotion_counts": per_emotion_total,
            },
            "detailed_results": detailed_results,
            "emotion_mapping": EMOTION_MAPPING,
            "model_info": {
                "model_name": ser_model_name_or_path,
                "available_emotions": list(label2id.keys()),
            }
        }, f, indent=2)

    print(f"Detailed results saved to: {results_file}")

    return {
        "ser_accuracy": accuracy * 100,
        "ser_avg_emotion_score": avg_emotion_score * 100,
        "ser_per_emotion_accuracy": per_emotion_accuracy,
        "ser_predictions": predictions,
        "ser_probabilities": probabilities,
        "ser_emotion_mapping": EMOTION_MAPPING,
        "ser_unmapped_emotion_stats": unmapped_stats,
        "ser_detailed_results": detailed_results,  # Add this for logging
    }
