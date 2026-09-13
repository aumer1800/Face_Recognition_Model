# Face Recognition Embedding Model 

## 1. Project Title

Face Recognition Embedding Model for an Attendance System

The project develops a deep learning-based face recognition component that can identify and verify students using their facial features.

The model is trained from scratch using the VGGFace2 dataset. Instead of directly predicting a person's name, the model learns a numerical representation called a face embedding.

Each face is converted into a 128-dimensional feature vector. These embeddings are compared using cosine similarity to determine whether two face images belong to the same person.

The trained model is intended to be integrated into a classroom attendance system.


## 2. Main Purpose of the Project

The main purpose of this project is to develop a reliable face recognition model that can be used for automatic attendance.

In a traditional attendance system, a teacher may need to manually mark each student's attendance.

With the proposed system:

Student
↓
Camera
↓
Face Detection
↓
Face Recognition Model
↓
128-Dimensional Face Embedding
↓
Cosine Similarity Comparison
↓
Student Identification
↓
Attendance Marked Automatically

The main objective is to reduce manual attendance work and provide a faster and more automated attendance process.


## 3. Main Technology Used

The project uses the following technologies and techniques:

Programming Language:
Python

Deep Learning Framework:
TensorFlow / Keras

Dataset:
VGGFace2

Dataset Source:
Kaggle — yakhyokhuja/vggface2-112x112

Model Type:
Custom CNN

Learning Approach:
Deep Metric Learning

Main Loss:
Batch-Hard Triplet Loss

Embedding Size:
128 dimensions

Similarity Method:
Cosine Similarity

Image Size:
112 × 112 pixels

Image Format:
RGB

Training Strategy:
Classification Warm-up + Triplet-Loss Training

Evaluation:
Face Verification using validation-selected threshold


# 4. Dataset

The model uses the VGGFace2 dataset.

VGGFace2 is a large-scale face recognition dataset containing images of many different people under different conditions.

The images can contain variations in:

- Facial expressions
- Head pose
- Lighting
- Age
- Camera conditions
- Background
- Image quality
- Face orientation

These variations are useful because a real attendance system will not always receive perfectly identical face images.


## 5. Dataset Configuration

Target identities:
8,000 people

Minimum images per identity:
40 images

Maximum images used per identity:
40 images

Image resolution:
112 × 112 pixels

Image channels:
3

Image format:
RGB

Total selected images:

8,000 × 40 = approximately 320,000 images


## 6. Dataset Split

The dataset is divided at the identity level.

80% of identities:
Training

10% of identities:
Validation

10% of identities:
Testing

For 8,000 identities:

Training identities:
6,400

Validation identities:
800

Test identities:
800


Approximate image distribution:

Training:
6,400 × 40 = 256,000 images

Validation:
800 × 40 = 32,000 images

Testing:
800 × 40 = 32,000 images

Total:
Approximately 320,000 images.


## 7. Why Identity-Level Splitting Is Used

The dataset is split by identity instead of randomly splitting images.

This is important for face recognition.

For example:

Training:
Person A
Person B
Person C

Validation:
Person D
Person E

Testing:
Person F
Person G

A person appearing in the training set does not appear in the validation or test sets.

This prevents the model from simply memorizing specific people.

It instead evaluates whether the learned facial representation can generalize to identities that were not seen during training.


# 8. Model Used

The main model used in this project is a Custom Convolutional Neural Network (CNN).

The CNN is trained from scratch.

This means the model does not depend on a pretrained face recognition backbone such as:

- FaceNet
- ArcFace
- VGGFace
- ResNet50 pretrained on ImageNet

Instead, the network learns facial features directly from the VGGFace2 training data.


## 9. Why a Custom CNN Is Used

A custom CNN is used because the main objective of the project is to develop and train a face embedding model from scratch.

The CNN learns hierarchical facial features.

At early layers, the model can learn simple features such as:

- Edges
- Lines
- Curves
- Basic textures

At deeper layers, it can learn more complex facial structures such as:

- Eyes
- Nose
- Mouth
- Face shape
- Facial proportions
- Higher-level identity-related patterns

The final layers convert these learned features into a compact 128-dimensional representation.


# 10. Face Embedding

The most important output of the model is a 128-dimensional face embedding.

For example:

Face Image
↓
CNN
↓
128-Dimensional Vector

The embedding can look conceptually like:

[0.12, -0.31, 0.08, 0.45, ... , -0.17]

There are 128 numerical values in total.

These numbers represent the learned characteristics of the face.

The system does not need to compare complete images pixel-by-pixel.

Instead, it compares their embeddings.


# 11. Why 128 Dimensions?

A 128-dimensional embedding provides a compact numerical representation of a face.

The goal is to make:

Images of the same person
→ Similar embeddings

Images of different people
→ Different embeddings

The 128-dimensional vector is small enough to store and compare efficiently while still providing enough capacity to represent useful facial information.


# 12. Main Training Pipeline

The complete training pipeline is:

VGGFace2 Dataset
↓
Identity Selection
↓
Identity-Based Train/Validation/Test Split
↓
112 × 112 Image Preprocessing
↓
Custom CNN
↓
Classification Warm-up
↓
Batch-Hard Triplet Loss
↓
128-Dimensional Face Embedding
↓
L2 Normalization
↓
Cosine Similarity
↓
Validation Threshold Selection
↓
Face Verification
↓
Final Model


# 13. Classification Warm-Up

Before using triplet loss, the CNN is first trained using a classification objective.

During this stage, the model learns to distinguish between the training identities.

For example:

Input Face
↓
CNN
↓
Feature Representation
↓
Identity Classification

This provides the CNN with a useful initial understanding of facial characteristics.

After the warm-up stage, the classification objective is replaced or supplemented by metric-learning training.


# 14. Batch-Hard Triplet Loss

The main metric-learning technique used in the project is Batch-Hard Triplet Loss.

Triplet learning uses three concepts:

Anchor
Positive
Negative

Anchor:
A face image of a particular person.

Positive:
Another face image of the same person.

Negative:
A face image belonging to a different person.


Example:

Anchor:
Student A - Image 1

Positive:
Student A - Image 2

Negative:
Student B - Image 1


The objective is:

Distance(Anchor, Positive)
should be small

Distance(Anchor, Negative)
should be large


In simple terms:

Same person
→ Pull embeddings closer

Different person
→ Push embeddings farther apart


# 15. Batch-Hard Concept

Instead of randomly selecting only one positive and one negative example, Batch-Hard Triplet Loss examines examples inside a training batch.

For each anchor, the model attempts to find:

Hardest Positive:
The same-person image that is most difficult to recognize as similar.

Hardest Negative:
The different-person image that is most similar to the anchor.

This forces the model to learn stronger discriminative facial features.


# 16. Cosine Similarity

After training, the system compares two face embeddings using cosine similarity.

If:

Embedding A = face of Student A

Embedding B = another image of Student A

then their cosine similarity should be high.

If:

Embedding A = Student A

Embedding B = Student B

then their cosine similarity should be lower.


Conceptually:

Same Person
→ High Similarity

Different Person
→ Low Similarity


# 17. L2 Normalization

The generated embeddings are L2-normalized before similarity comparison.

Normalization helps make the embedding vectors comparable and makes cosine similarity more stable.

The normalized embedding has a controlled magnitude, while its direction contains the important identity information used for comparison.


# 18. Verification Threshold

The model does not simply say:

"Similarity > 0 = same person"

Instead, a threshold is selected using the validation dataset.

For example:

Cosine similarity >= threshold
→ Same person

Cosine similarity < threshold
→ Different person

The actual threshold is selected experimentally from validation data rather than being arbitrarily chosen.


# 19. Why Validation Threshold Selection Is Important

Different models produce different similarity distributions.

Therefore, one fixed threshold may not work equally well for every model.

The validation set is used to test multiple possible thresholds.

The threshold that provides the best verification performance is selected.

The test set is then used for final evaluation.

This prevents using the test data to tune the threshold.


# 20. Main Features of the Model

The major features of the proposed model are:

1. Face image input

2. 112 × 112 RGB image processing

3. Custom CNN trained from scratch

4. Identity-based dataset splitting

5. Classification warm-up

6. Batch-Hard Triplet Loss

7. 128-dimensional face embeddings

8. L2 embedding normalization

9. Cosine similarity comparison

10. Validation-based threshold selection

11. Face verification

12. Model saving for later deployment

13. Designed for integration with an attendance system


# 21. Important Parameters

The main parameters used in the project include:

Dataset:
VGGFace2

Target identities:
8,000

Minimum images per identity:
40

Maximum images per identity:
40

Image size:
112 × 112

Color format:
RGB

Training split:
80%

Validation split:
10%

Testing split:
10%

Embedding dimension:
128

Loss:
Batch-Hard Triplet Loss

Training strategy:
Classification warm-up followed by metric learning

Similarity:
Cosine similarity

Embedding normalization:
L2 normalization

Threshold:
Selected using validation data

Model:
Custom CNN

Training:
From scratch


# 22. Data Preprocessing

Before an image is passed to the model, it is processed into the required format.

The general preprocessing pipeline is:

Original Image
↓
Load Image
↓
Resize to 112 × 112
↓
Convert to RGB
↓
Normalize Pixel Values
↓
Convert to Tensor
↓
CNN


This ensures that all images have a consistent input format.


# 23. Data Augmentation

Data augmentation can be used during training to make the model more robust.

Possible augmentation techniques include:

- Random horizontal flipping
- Small rotations
- Small translations
- Zooming
- Brightness changes
- Contrast changes

The purpose is to simulate variations that can occur when the attendance camera captures students under different conditions.

Augmentation should normally be applied to training images, while validation and test images should remain consistent for fair evaluation.


# 24. What the Model Learns

The model does not simply memorize the person's name.

It learns facial characteristics that can be represented numerically.

These characteristics may include:

- Eye structure
- Distance between facial landmarks
- Nose shape
- Mouth structure
- Face proportions
- Jaw structure
- Overall facial geometry
- Local facial textures
- Higher-level identity patterns

The CNN combines these low-level and high-level features into the final 128-dimensional embedding.


# 25. Training Objective

The overall objective is to create an embedding space where:

Same identity:
Embeddings are close together.

Different identities:
Embeddings are far apart.


Conceptually:

           Same Person
        ●    ●    ●
          ●  ●
           ↓
      Similar Embeddings


Different Person

        ●                         ●
     Student A                 Student B

      Different Embeddings


# 26. Face Verification vs Face Classification

This project primarily uses face verification rather than traditional classification.

Classification asks:

"Which known class/person is this?"

Verification asks:

"Are these two faces the same person?"

This is useful for an attendance system because new students can potentially be registered by generating and storing their embeddings.

The system can then compare a newly captured face against stored student embeddings.


# 27. Attendance System Integration

The trained embedding model can be integrated into the attendance system using the following workflow:

Student Registration
↓
Capture Student Face
↓
Generate 128-D Embedding
↓
Store Student Information + Embedding
↓
Create Course
↓
Enroll Student
↓
Start Attendance Session
↓
Open Camera
↓
Capture Face
↓
Generate Embedding
↓
Compare with Registered Embeddings
↓
Identify Student
↓
Check Course Enrollment
↓
Check Duplicate Attendance
↓
Mark Student Present
↓
Generate Attendance Report


# 28. Example of Attendance Recognition

Suppose three students are registered:

Student A
Student B
Student C

Their face embeddings are stored.

During attendance, the camera captures a face.

The model generates:

Live Face
→ 128-D embedding

The system compares this embedding against:

Student A embedding
Student B embedding
Student C embedding

If the similarity with Student B is sufficiently high and passes the threshold:

Student B
→ Recognized
→ Check enrollment
→ Check duplicate attendance
→ Mark Present


# 29. Main Advantages

The proposed approach provides several advantages:

1. It uses a large-scale face dataset.

2. It uses identity-level data splitting.

3. It learns embeddings instead of relying only on classification.

4. The 128-D embeddings are compact and efficient to store.

5. Cosine similarity provides a simple method for face comparison.

6. Batch-Hard Triplet Loss improves separation between identities.

7. Validation-based threshold selection provides a data-driven verification threshold.

8. The model can potentially support new registered students without retraining the entire recognition model.

9. The embedding approach is suitable for integration with an attendance database.

10. The model is trained from scratch, making the project suitable for demonstrating the complete deep learning pipeline.


# 30. Limitations

Although the model is designed for face recognition, several limitations may remain.

Performance can be affected by:

- Poor lighting
- Very low-resolution camera input
- Extreme head angles
- Occlusion
- Masks
- Glasses
- Motion blur
- Large distance from the camera
- Multiple faces appearing simultaneously
- Similar-looking individuals
- Poor face alignment
- Insufficient registration images


Another limitation is that a custom CNN trained from scratch may not achieve the same recognition performance as modern large-scale pretrained face recognition models.


# 31. Future Improvements

The project can be improved in several ways.

### 1. Use a Stronger Backbone

The custom CNN can later be replaced or compared with stronger architectures such as:

- ResNet50
- ResNet100
- EfficientNet
- MobileFaceNet
- Vision Transformer-based models

This could improve recognition accuracy.


### 2. ArcFace Loss

Batch-Hard Triplet Loss can be compared with ArcFace.

ArcFace introduces an angular margin that can produce highly discriminative face embeddings.

A future version could compare:

Custom CNN + Triplet Loss

versus

ResNet + ArcFace


### 3. Larger Dataset

The model can be trained using more identities and more images per identity if sufficient GPU resources are available.

This may improve generalization.


### 4. Better Face Detection

A dedicated face detector can be added before recognition.

For example:

Camera
↓
Face Detection
↓
Face Alignment
↓
Face Recognition
↓
Attendance


This prevents the recognition network from receiving unnecessary background information.


### 5. Face Alignment

Faces can be aligned using facial landmarks.

For example:

Eyes
Nose
Mouth

can be used to normalize the orientation of the face before recognition.


### 6. Multiple Images Per Student

Instead of storing only one face embedding for each student, multiple embeddings can be stored.

For example:

Student A

Registration Image 1
Registration Image 2
Registration Image 3
Registration Image 4
Registration Image 5


This can improve recognition under different conditions.


### 7. Real-Time Recognition

The final system can process camera frames continuously.

Camera
↓
Frame
↓
Face Detection
↓
Embedding Generation
↓
Similarity Search
↓
Student Recognition


The system can then mark attendance automatically.


### 8. Multiple-Face Recognition

The system can be extended to recognize multiple students in the same camera frame.

For example:

Classroom Camera
↓
Face 1 → Student A
Face 2 → Student B
Face 3 → Student C
Face 4 → Student D


All recognized and enrolled students can then be marked present.


### 9. Better Database Integration

The embedding system can be connected to PostgreSQL.

The database can store:

Student information
Face embeddings
Courses
Course enrollment
Class schedules
Attendance sessions
Attendance records


This makes the model part of a complete attendance management system.


### 10. Anti-Spoofing / Liveness Detection

A future version should include liveness detection.

Without liveness detection, someone might attempt to present:

- A photograph
- A phone screen
- A video

to the camera.

A liveness model could determine whether the detected face belongs to a real person who is physically present.


### 11. Threshold Optimization

Future work can investigate different threshold-selection strategies.

Possible evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- False Acceptance Rate (FAR)
- False Rejection Rate (FRR)
- Equal Error Rate (EER)


### 12. Real-World Classroom Testing

The model should eventually be tested using real classroom conditions.

Testing should include:

- Different lighting
- Different camera positions
- Different student distances
- Different facial expressions
- Multiple students
- Partial occlusion
- Different backgrounds


# 32. Expected Final System

The complete final system can be represented as:

                CLASSROOM CAMERA
                       ↓
                Face Detection
                       ↓
                 Face Alignment
                       ↓
              112 × 112 Face Image
                       ↓
                  Custom CNN
                       ↓
             128-D Face Embedding
                       ↓
                L2 Normalization
                       ↓
              Cosine Similarity
                       ↓
              Threshold Checking
                       ↓
              Student Recognition
                       ↓
             Enrollment Verification
                       ↓
             Duplicate Attendance Check
                       ↓
               Mark Attendance
                       ↓
              Attendance Database
                       ↓
                Reports / History


# 33. Overall Project Contribution

The main contribution of this project is the development of a complete face embedding pipeline for an automated attendance system.

Instead of depending only on direct image classification, the system learns a compact 128-dimensional representation of faces.

The combination of:

VGGFace2
+
Identity-Level Splitting
+
Custom CNN
+
Classification Warm-up
+
Batch-Hard Triplet Loss
+
128-D Embeddings
+
L2 Normalization
+
Cosine Similarity
+
Validation Threshold Selection

provides a complete deep learning-based face verification pipeline.


# 34. Final Summary

This project develops a face recognition embedding model trained from scratch using the VGGFace2 dataset.

The selected dataset contains up to 8,000 identities with a maximum of 40 images per identity, resulting in approximately 320,000 images.

The images are divided using an identity-level split of 80% training, 10% validation, and 10% testing.

A Custom CNN is used to learn facial features. The model first undergoes classification warm-up and is then trained using Batch-Hard Triplet Loss.

The final network generates a 128-dimensional face embedding for each input face.

The embeddings are L2-normalized and compared using cosine similarity.

A verification threshold is selected using the validation dataset.

The final model is intended to be integrated into a classroom attendance system where a camera captures students' faces, generates embeddings, compares them against registered student embeddings, verifies the identity, and automatically records attendance.

Future improvements can include stronger CNN architectures, ArcFace loss, improved face detection and alignment, multiple face recognition, real-time optimization, liveness detection, larger datasets, and extensive real-world classroom testing.
