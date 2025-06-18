# Rinda Project: Plant Disease Classification using Machine Learning

### Problem Statement

Smallholder farmers in Rwanda face significant and preventable crop losses, estimated to be between 26% and 36% for key staple crops like potatoes and cassava, primarily due to pests and diseases. This issue directly threatens the livelihoods of a majority of the population and undermines national food security. The current methods for identifying and managing these crop health threats rely on manual inspection and traditional extension services, which are often too slow, inaccessible, or lack the diagnostic accuracy needed for timely and effective intervention. This results in delayed treatment, reduced yields, and inefficient use of resources, creating a critical need for a more accessible and precise decision-making tool for farmers.

### **Project Overview**

This project explores the implementation and optimization of machine learning models for a multi-class image classification task. The objective is to accurately identify various diseases in plants from leaf images. We compare the performance of a classical algorithm (Support Vector Machine) against several iterations of a Convolutional Neural Network (CNN), applying various optimization techniques to improve the model's accuracy, generalization, and efficiency.

---

### **Dataset**

The dataset used for this project is the **PlantVillage Dataset**, a rich public collection of plant leaf images.

* **Source:** [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
* **Characteristics:**
    * **Volume:** Contains over 54,000 images.
    * **Variety:** The full dataset includes 38 distinct classes representing different plant species and their diseases (or healthy status). For this project, a subset of 6 classes was used to manage computational resources.
    * **Challenge:** 
      * During Exploratory Data Analysis (EDA), a significant **class imbalance** was identified, with some classes having far more images than others. This was a key consideration during model training, as it can bias the model towards majority classes.
      * The Dataset is also enormous which caused the **initial training to last more than 20 hours**, this led to the decision of reducing the dataset to plants and diseases that are most relevant to Rwanda.

---

### **Getting Started**

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

#### **Prerequisites**

You will need Python 3 and the following libraries installed:
* TensorFlow
* Scikit-learn
* NumPy
* Seaborn
* Matplotlib
* Joblib

You can install them using pip:
```bash
pip install tensorflow scikit-learn numpy seaborn matplotlib joblib
```

#### **Setup and Execution**

1.  **Clone the repository:**
    ```bash
    git clone [the repository](https://github.com/Christianib003/rinda-project)
    cd rinda-project
    ```

2.  **Run the Notebook:**
    * Open and run the `notebook.ipynb` file in a Jupyter environment. The notebook contains all the code for data preprocessing, model training, and evaluation.

---

### **Implementation Choices & Methodology**

The project compares two main approaches: a classical machine learning algorithm and a deep learning approach using Neural Networks.

#### **1. Classical Model: Support Vector Machine (SVM)**

* **Choice Rationale:** SVM was chosen as the classical algorithm because it is a powerful and versatile classifier. To make it comparable to a CNN, a feature engineering step was required.
* **Methodology:**
    1.  **Feature Extraction:** Since SVMs cannot process raw image data directly, each image was converted to a 1D feature vector by flattening its pixel values.
    2.  **Preprocessing:** `StandardScaler` was applied to normalize the feature values, which is crucial for the performance of distance-based algorithms like SVM.
    3.  **Dimensionality Reduction:** To manage the high dimensionality (16,384 features per image) and significantly reduce training time, **Principal Component Analysis (PCA)** was used to reduce the features to the 150 most important components.
    4.  **Hyperparameter Tuning:** `RandomizedSearchCV` was used on a subset of the data to efficiently find the optimal hyperparameters (`C` and `gamma`) without the excessive computational cost of a full `GridSearchCV`.

#### **2. Neural Network Models (CNN)**

* **Choice Rationale:** Convolutional Neural Networks (CNNs) are the state-of-the-art for image classification tasks as they can automatically learn spatial hierarchies of features (from edges to complex patterns).
* **Baseline Model Architecture:**
    * **Input Layer:** `(128, 128, 3)` to match the resized image dimensions.
    * **Convolutional Layers:** Two `Conv2D` layers with `MaxPooling2D` to extract features.
    * **Dense Layer:** A fully connected `Dense` layer with 128 neurons.
    * **Output Layer:** A `Dense` layer with 6 neurons (for 6 classes) using a `softmax` activation function.
* **Optimization Techniques Explored:**
    * **Optimizers:** Tested `Adam`, `RMSprop`, and `SGD` to observe their impact on convergence and performance.
    * **Regularization:** Applied `L1`, `L2`, and `Dropout` regularization to combat the overfitting observed in the baseline model.
    * **Early Stopping:** Used to monitor validation loss and prevent the model from training unnecessarily long after performance has peaked.
    * **Learning Rate:** Experimented with different learning rates to fine-tune the model's convergence.

#### **Code Modularity**

To adhere to the DRY (Don't Repeat Yourself) principle and ensure the code is maintainable, several reusable functions were created in the notebook:
* `split_dataset()`: To partition a TensorFlow dataset.
* `calculate_metrics()`: To compute and return a dictionary of performance metrics.
* `plot_training_history()`: To visualize the model's learning curves.
* `build_cnn_model()`: A flexible function to build CNNs with varying optimization hyperparameters.

---

### **Results and Discussion**

#### **Neural Network Experiments Table**

The following table details the 5 training instances required by the assignment, showing the different combinations of hyperparameters used for the CNN and the resulting performance on the **validation set**.

| Training Instance | Optimizer | Learning Rate | Dropout Rate | Regularizer Used | Epochs (Stopped) | Number of Dense Layers | Validation Accuracy | Validation F1-score | Validation Recall | Validation Precision |
| :--- | :--- | :--- | :--- | :--- | :--- |:--- |:--- |:--- |:--- |:--- |
| **Instance 1 (Baseline)** | RMSprop (default) | 0.001 | 0.0 | None | 15 (Fixed) | 1 | 71.13% | 64.67% | 66.22% | 71.90% |
| **Instance 2** | Adam | 0.001 | 0.5 | L2 (0.01) | 34 | 1 | 95.49% | 86.08% | 85.04% | 89.67% |
| **Instance 3** | RMSprop | 0.001 | 0.2 | L1 (0.01) | 7 | 1 | 54.08% | 42.02% | 45.02% | 43.33% |
| **Instance 4** | SGD | 0.001 | 0.4 | None | 22 | 1 | 31.41% | 7.97% | 16.67% | 5.23% |
| **Instance 5 (Best)** | **Adam** | **0.0001** | **0.5** | **L2 (0.01)** | **39** | **1** | **96.20%** | **85.38%** | **83.68%** | **97.14%** |

*Note: The "Number of Dense Layers" column refers to the hidden (fully connected) layers in the architecture, which was kept constant at 1 layer with 128 neurons for these experiments.*


#### **Analysis of Results & Key Findings**

* **Baseline vs. Optimized:** The baseline model (Instance 1) showed significant **overfitting**, with training accuracy far exceeding validation accuracy. Instance 2, which introduced `Dropout`, `L2 Regularization`, and `EarlyStopping`, dramatically reduced overfitting and improved accuracy by over 24%. This clearly demonstrates the effectiveness of these optimization techniques.

* **Hyperparameter Impact:** The choice of optimizer and regularizer had a major impact. The combination of `RMSprop` with a strong `L1` penalty (Instance 3) and the standard `SGD` optimizer (Instance 4) both failed to converge and resulted in very poor performance. This highlights that not all optimization strategies are suitable for every problem. The `Adam` optimizer proved to be the most effective for this task.

* **Fine-Tuning for Best Performance:** Instance 5 was a fine-tuning of our best model (Instance 2). By reducing the learning rate from `0.001` to `0.0001`, the model was able to converge more precisely, achieving the **highest validation accuracy of 96.20%**.

---

### **Final Model Comparison**

To provide a final, unbiased evaluation, the best models were tested on the held-out **test set**.

| Model | Test Accuracy | Test F1-score | Test Precision | Test Recall |
| :--- | :--- | :--- | :--- | :--- |
| **Tuned SVM** | 49.15% | 34.84% | 47.87% | 38.72% |
| **Best CNN (Instance 5)** | **96.20%** | **85.38%** | **97.14%** | **83.68%** |


#### **Conclusion: Which implementation was better?**

The **Convolutional Neural Network (CNN) was overwhelmingly superior** to the Support Vector Machine (SVM).

* The best CNN achieved a test accuracy of **96.2%**, more than double the SVM's accuracy of **49.15%**.
* This result is expected for image classification tasks. CNNs are specifically designed to learn spatial features like textures, shapes, and patterns directly from image pixels. In contrast, the classical SVM, even with preprocessing, struggles to interpret the high-dimensional feature space of flattened images effectively.

---

### **How to Load the Best Model**

The best-performing model (Instance 5, saved as `optimized_cnn_model_4.keras`) can be loaded and used for predictions with the following Python code:

```python
import tensorflow as tf

# Load the model
best_model = tf.keras.models.load_model('saved_models/optimized_cnn_model_4.keras')

# Display the model's architecture
best_model.summary()

# You can now use this model to make predictions on new images
# predictions = best_model.predict(new_image_data)
