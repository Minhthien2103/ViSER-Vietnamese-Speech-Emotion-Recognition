import os
import time
import re
import json
import shutil
import subprocess
import pandas as pd
import torch
import torchaudio
import warnings

# Tắt các cảnh báo lặt vặt của PyTorch cho terminal gọn gàng
warnings.filterwarnings("ignore")

# ==========================================
# CẤU HÌNH THÔNG SỐ HỆ THỐNG
# ==========================================
# 1. Thư mục và File
RAW_AUDIO_DIR = "raw_audio"               # Chứa audio gốc tải từ yt-dlp
CHUNKS_DIR = "processed_chunks"           # Chứa các đoạn ngắn xuất ra
FINAL_DATASET_DIR = "dataset_final"       # Chứa dataset hoàn chỉnh sau khi sắp xếp
PRE_LABEL_CSV = "dataset_prelabel.csv"    # File trung gian nạp vào Label Studio
FINAL_JSON_METADATA = "SENSE_metadata.json" # File JSON đích cuối cùng

# 2. API & Cảm xúc
GROQ_API_KEY = "ĐIỀN_API_KEY_CỦA_BẠN_VÀO_ĐÂY"
EMOTION_MAP = {
    "Happy": 0,
    "Sad": 1,
    "Angry": 2,
    "Neutral": 3
}

# 3. Thông số xử lý âm thanh
SAMPLE_RATE = 16000                   
MIN_SEC = 3.0                         
MAX_SEC = 5.0                         

# ==========================================
# MODULE 1: DOWNLOAD AUDIO (YT-DLP)
# ==========================================
def run_download_audio():
    csv_path = input("Nhập đường dẫn file CSV chứa link (vd: sources.csv): ").strip()
    
    if not os.path.exists(csv_path):
        print(f"❌ Không tìm thấy file {csv_path}. Vui lòng kiểm tra lại!")
        return

    df = pd.read_csv(csv_path)
    total_videos = len(df)
    print(f"🚀 Bắt đầu tiến trình tải {total_videos} video...\n" + "-"*40)
    
    for index, row in df.iterrows():
        url = str(row['url']).strip()
        source_name = str(row['source_name']).strip()
        region = str(row['expected_region']).strip() 
        
        output_dir = os.path.join(RAW_AUDIO_DIR, region)
        os.makedirs(output_dir, exist_ok=True)
        
        file_name = f"{region}_{source_name}"
        output_template = os.path.join(output_dir, f"{file_name}.%(ext)s")
        
        print(f"⏳ [{index+1}/{total_videos}] Đang xử lý: {source_name} (Vùng: {region})")
        
        command = [
            "yt-dlp",
            "-x", 
            "--audio-format", "wav", 
            "--postprocessor-args", "-ar 16000 -ac 1", 
            "-o", output_template, 
            url
        ]
        
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            print(f"✅ Thành công: Đã lưu vào {output_dir}\n")
        except subprocess.CalledProcessError:
            print(f"❌ THẤT BẠI: Lỗi khi tải video {url}. Bỏ qua và tải video tiếp theo.\n")
        except Exception as e:
            print(f"⚠️ Lỗi không xác định: {e}\n")

    print("-" * 40 + "\n🎉 HOÀN TẤT QUÁ TRÌNH TẢI DỮ LIỆU!")

# ==========================================
# MODULE 2: VAD CHUNKING (CẮT AUDIO)
# ==========================================
def run_vad_chunking():
    if not os.path.exists(RAW_AUDIO_DIR):
        print(f"❌ Không tìm thấy thư mục gốc '{RAW_AUDIO_DIR}'! Bạn cần chạy Bước 1 trước.")
        return

    print("⏳ Đang tải mô hình Silero VAD...")
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        trust_repo=True
    )
    get_speech_timestamps = utils[0]
    print("✅ Tải mô hình thành công!\n" + "="*40)

    total_files_processed = 0
    total_chunks_created = 0

    for root, dirs, files in os.walk(RAW_AUDIO_DIR):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                
                relative_path = os.path.relpath(root, RAW_AUDIO_DIR)
                current_output_dir = os.path.join(CHUNKS_DIR, relative_path)
                os.makedirs(current_output_dir, exist_ok=True)
                
                file_prefix = os.path.splitext(file)[0]
                print(f"Đang cắt file: {file_prefix}...")
                
                try:
                    wav, sr = torchaudio.load(file_path)
                    if sr != SAMPLE_RATE:
                        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
                    if wav.shape[0] > 1:
                        wav = torch.mean(wav, dim=0, keepdim=True)

                    wav_1d = wav.squeeze(0) 
                    speech_timestamps = get_speech_timestamps(wav_1d, model, sampling_rate=SAMPLE_RATE)
                    
                    chunk_idx = 1
                    saved_chunks = 0
                    
                    for ts in speech_timestamps:
                        start_frame = ts['start']
                        end_frame = ts['end']
                        current_start = start_frame
                        
                        while (end_frame - current_start) >= int(MIN_SEC * SAMPLE_RATE):
                            current_end = min(current_start + int(MAX_SEC * SAMPLE_RATE), end_frame)
                            chunk_wav = wav[:, current_start:current_end]
                            
                            out_name = f"{file_prefix}_chunk{chunk_idx:04d}.wav"
                            out_path = os.path.join(current_output_dir, out_name)
                            
                            torchaudio.save(out_path, chunk_wav, SAMPLE_RATE)
                            chunk_idx += 1
                            saved_chunks += 1
                            current_start = current_end

                    print(f"  👉 Cắt thành công {saved_chunks} đoạn (3-5s).")
                    total_files_processed += 1
                    total_chunks_created += saved_chunks

                except Exception as e:
                    print(f"❌ Lỗi xử lý file {file_prefix}: {e}")

    print("="*40)
    print(f"🎉 HOÀN TẤT CHUNKING! Tổng số file gốc đã xử lý: {total_files_processed}")
    print(f"Tổng số audio chunks tạo ra: {total_chunks_created}")

# ==========================================
# MODULE 3: AI AUTO-LABELING (WHISPER + GROQ)
# ==========================================
def run_auto_labeling():
    from faster_whisper import WhisperModel
    from groq import Groq

    if not os.path.exists(CHUNKS_DIR):
        print(f"❌ Không tìm thấy thư mục '{CHUNKS_DIR}'! Bạn cần chạy Bước 2 trước.")
        return

    groq_client = Groq(api_key=GROQ_API_KEY)
    print("⏳ Đang tải mô hình Whisper...")
    whisper_model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
    
    results = []
    
    for root, dirs, files in os.walk(CHUNKS_DIR):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                accent = os.path.basename(root)
                
                print(f"Đang xử lý: {accent}/{file}")
                try:
                    segments, _ = whisper_model.transcribe(file_path, language="vi")
                    text_for_llm = " ".join([s.text for s in segments]).strip()
                except:
                    text_for_llm = ""

                prompt = f'Chỉ trả về 1 từ duy nhất (Happy, Sad, Angry, Neutral) mô tả cảm xúc câu này: "{text_for_llm}"'
                try:
                    chat = groq_client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="llama-3.3-70b-versatile",
                        temperature=0, max_tokens=10
                    )
                    emotion_ai = re.sub(r'[^A-Za-z]', '', chat.choices[0].message.content.strip().capitalize())
                    if emotion_ai not in EMOTION_MAP: emotion_ai = "Neutral"
                except:
                    emotion_ai = "Neutral"

                results.append({
                    "file_path": file_path,
                    "accent": accent,
                    "emotion_ai": emotion_ai
                })
                time.sleep(1.5) 

    df = pd.DataFrame(results)
    df.to_csv(PRE_LABEL_CSV, index=False, encoding='utf-8-sig')
    print(f"✅ Xong! Hãy đưa file '{PRE_LABEL_CSV}' lên Label Studio.")

# ==========================================
# MODULE 4: POST-PROCESSING & JSON BUILDER
# ==========================================
def clean_ls_label(raw_label):
    match = re.search(r'([a-zA-Z_]+)', str(raw_label))
    return match.group(1) if match else "Unknown"

def run_post_processing():
    export_csv = input("Nhập tên file CSV tải từ Label Studio về (vd: project_export.csv): ").strip()
    if not os.path.exists(export_csv):
        print("❌ Không tìm thấy file CSV!")
        return

    df = pd.read_csv(export_csv)
    final_metadata = []
    
    print("🚀 Đang copy file và tạo Metadata JSON...")
    
    for index, row in df.iterrows():
        original_path = str(row.get('file_path', '')).replace("http://localhost:8081/", "")
        if not os.path.exists(original_path):
            continue

        emotion = clean_ls_label(row.get('final_emotion', ''))
        accent = clean_ls_label(row.get('final_region', ''))
        gender = clean_ls_label(row.get('final_gender', ''))

        if accent == "Trash" or emotion == "Unknown" or gender == "Unknown":
            continue

        filename = os.path.basename(original_path)
        speaker_id = filename.replace(".wav", "").split("_chunk")[0].replace(accent + "_", "")

        waveform, sample_rate = torchaudio.load(original_path)
        duration = round(waveform.shape[-1] / sample_rate, 2)

        target_folder = os.path.join(FINAL_DATASET_DIR, gender, accent, emotion)
        os.makedirs(target_folder, exist_ok=True)
        target_path = os.path.join(target_folder, filename)
        
        shutil.copy2(original_path, target_path)

        final_metadata.append({
            "speaker_id": speaker_id,
            "path": target_path.replace("\\", "/"), 
            "duration": duration,
            "accent": accent,
            "emotion": emotion,
            "emotion_id": EMOTION_MAP.get(emotion, 3),
            "gender": gender
        })

    with open(FINAL_JSON_METADATA, 'w', encoding='utf-8') as f:
        json.dump(final_metadata, f, ensure_ascii=False, indent=4)
        
    print(f"🎉 HOÀN TẤT! Copy thành công {len(final_metadata)} files.")
    print(f"📂 File Metadata tổng hợp được lưu tại: '{FINAL_JSON_METADATA}'")

# ==========================================
# ĐIỀU HƯỚNG MENU CHÍNH
# ==========================================
if __name__ == "__main__":
    while True:
        print("\n" + "="*50)
        print("🚀 HỆ THỐNG XỬ LÝ DỮ LIỆU ÂM THANH S.E.N.S.E")
        print("="*50)
        print("1. Tải Audio từ YouTube (Dựa trên file CSV)")
        print("2. Cắt nhỏ Audio (Silero VAD - 3s đến 5s)")
        print("3. AI Gán nhãn sơ bộ (Tạo file cho Label Studio)")
        print("4. Hậu xử lý (Copy File chuẩn & Tạo JSON Metadata)")
        print("0. Thoát chương trình")
        print("="*50)
        
        choice = input("👉 Nhập số chức năng bạn muốn chạy (0-4): ").strip()
        
        if choice == '1':
            run_download_audio()
        elif choice == '2':
            run_vad_chunking()
        elif choice == '3':
            run_auto_labeling()
        elif choice == '4':
            run_post_processing()
        elif choice == '0':
            print("👋 Đã thoát chương trình. Hẹn gặp lại!")
            break
        else:
            print("⚠️ Lựa chọn không hợp lệ. Vui lòng nhập từ 0 đến 4!")