import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os
from datetime import datetime
import time

class FaceRecognitionSystem:
    def __init__(self):
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.knn_classifier = None  # Akan diinisialisasi saat training
        self.face_database = []
        self.labels = []
        
    def extract_face_features(self, image):
        """Mengekstrak fitur wajah dari gambar"""
        if image is None:
            return None
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None
        
        (x, y, w, h) = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (100, 100))
        return face_roi.flatten()
    
    def train_model(self, dataset_path):
        """Melatih model KNN dengan dataset wajah"""
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path '{dataset_path}' tidak ditemukan!")
            
        print("Memulai proses pelatihan...")
        total_images = 0
        
        # Reset database
        self.face_database = []
        self.labels = []
        
        for person_id in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person_id)
            if os.path.isdir(person_path):
                image_count = 0
                for image_file in os.listdir(person_path):
                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(person_path, image_file)
                        try:
                            image = cv2.imread(image_path)
                            if image is None:
                                print(f"Gagal membaca gambar: {image_path}")
                                continue
                                
                            features = self.extract_face_features(image)
                            if features is not None:
                                self.face_database.append(features)
                                self.labels.append(person_id)
                                image_count += 1
                                total_images += 1
                            else:
                                print(f"Tidak ada wajah terdeteksi dalam gambar: {image_path}")
                        except Exception as e:
                            print(f"Error memproses gambar {image_path}: {str(e)}")
                
                print(f"Berhasil memproses {image_count} gambar untuk {person_id}")
        
        if total_images == 0:
            raise ValueError("Tidak ada gambar wajah valid yang ditemukan di dataset!")
            
        # Konversi list ke numpy array
        X = np.array(self.face_database)
        y = np.array(self.labels)
        
        # Tentukan jumlah tetangga berdasarkan jumlah data
        n_neighbors = min(5, total_images)
        print(f"Menggunakan {n_neighbors} tetangga untuk klasifikasi")
        
        # Inisialisasi KNN dengan jumlah tetangga yang sesuai
        self.knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        
        # Training model
        print(f"Total data training: {len(X)} gambar")
        self.knn_classifier.fit(X, y)
        print("Pelatihan selesai!")
    
    def recognize_face(self, image):
        """Mengenali wajah dari gambar"""
        if self.knn_classifier is None:
            print("Model belum dilatih! Harap latih model terlebih dahulu.")
            return None
            
        features = self.extract_face_features(image)
        if features is not None:
            features = features.reshape(1, -1)  # Reshape untuk prediksi
            prediction = self.knn_classifier.predict(features)
            return prediction[0]
        return None
    
    def record_attendance(self, person_id, image):
        """Mencatat kehadiran"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        folder_path = os.path.join("attendance_images", person_id)
        os.makedirs(folder_path, exist_ok=True)
        image_filename = os.path.join(folder_path, f"{timestamp.replace(':', '-')}.jpg")
        cv2.imwrite(image_filename, image)
        with open("attendance.txt", "a") as f:
            f.write(f"{person_id},{timestamp}\\n")
        print(f"Kehadiran tercatat: {person_id} pada {timestamp}")
        print(f"Gambar disimpan di folder: {folder_path}")

def check_dataset(dataset_path):
    """Mengecek dan memvalidasi dataset"""
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        print(f"Folder {dataset_path} telah dibuat.")
        print("Silakan ikuti langkah berikut:")
        print("1. Buat subfolder untuk setiap orang (misal: person1, person2, dll)")
        print("2. Tambahkan minimal 1 foto wajah per orang")
        print("3. Jalankan program kembali")
        return False
        
    # Hitung jumlah gambar valid
    total_images = 0
    for person_id in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_id)
        if os.path.isdir(person_path):
            images = [f for f in os.listdir(person_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            total_images += len(images)
    
    if total_images == 0:
        print("Dataset kosong atau tidak valid!")
        print("Silakan tambahkan foto wajah ke folder dataset_wajah/")
        return False
    
    print(f"Ditemukan {total_images} gambar dalam dataset")
    return True

def main():
    try:
        # Inisialisasi sistem
        system = FaceRecognitionSystem()
        
        # Cek dataset
        dataset_path = "dataset_wajah"
        if not check_dataset(dataset_path):
            return
            
        # Melatih model dengan dataset
        system.train_model(dataset_path)
        
        # Memulai capture video
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Tidak dapat mengakses kamera!")
        
        print("\nTekan 'q' untuk keluar dari program")
        countdown_start = None
        recognized_person = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Gagal membaca frame dari kamera")
                break
                
            # Mendeteksi dan mengenali wajah
            person_id = system.recognize_face(frame)
            
            if person_id is not None:
                if recognized_person == person_id:
                    if countdown_start is None:
                        countdown_start = time.time()
                        print(f"Mulai countdown untuk {person_id}")
                    else:
                        elapsed = int(5 - (time.time() - countdown_start))
                        if elapsed > 0:
                            cv2.putText(frame, f"Mencatat dalam {elapsed} detik...", (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        else:
                            system.record_attendance(person_id, frame)
                            break  # Keluar setelah mencatat kehadiran
                else:
                    recognized_person = person_id
                    countdown_start = None  # Reset countdown jika wajah baru dikenali
            else:
                recognized_person = None
                countdown_start = None
                cv2.putText(frame, "Tidak ada wajah yang dikenali!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Menampilkan frame
            if recognized_person:
                cv2.putText(frame, f"ID: {recognized_person}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Face Recognition Attendance", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Terjadi error: {str(e)}")
        
if __name__ == "__main__":
    main()
