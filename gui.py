import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

class BreastCancerPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Breast Cancer Prediction System")
        self.root.geometry("1000x800")
        self.root.configure(bg='#f0f0f0')
        
        # Feature names for the 30 features
        self.feature_names = [
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
            'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
            'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
            'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
            'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
            'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
        ]
        
        # Feature descriptions
        self.feature_descriptions = {
            'radius_mean': 'Mean distance from center to perimeter',
            'texture_mean': 'Mean standard deviation of gray-scale values',
            'perimeter_mean': 'Mean perimeter of the nucleus',
            'area_mean': 'Mean area of the nucleus',
            'smoothness_mean': 'Mean local variation in radius lengths',
            'compactness_mean': 'Mean perimeter^2 / area - 1.0',
            'concavity_mean': 'Mean severity of concave portions',
            'concave points_mean': 'Mean number of concave portions',
            'symmetry_mean': 'Mean symmetry of the nucleus',
            'fractal_dimension_mean': 'Mean coastline approximation - 1',
            'radius_se': 'Standard error of radius',
            'texture_se': 'Standard error of texture',
            'perimeter_se': 'Standard error of perimeter',
            'area_se': 'Standard error of area',
            'smoothness_se': 'Standard error of smoothness',
            'compactness_se': 'Standard error of compactness',
            'concavity_se': 'Standard error of concavity',
            'concave points_se': 'Standard error of concave points',
            'symmetry_se': 'Standard error of symmetry',
            'fractal_dimension_se': 'Standard error of fractal dimension',
            'radius_worst': 'Worst (largest) radius value',
            'texture_worst': 'Worst texture value',
            'perimeter_worst': 'Worst perimeter value',
            'area_worst': 'Worst area value',
            'smoothness_worst': 'Worst smoothness value',
            'compactness_worst': 'Worst compactness value',
            'concavity_worst': 'Worst concavity value',
            'concave points_worst': 'Worst concave points value',
            'symmetry_worst': 'Worst symmetry value',
            'fractal_dimension_worst': 'Worst fractal dimension value'
        }
        
        # Load model components if available
        self.load_model()
        
        # Create GUI
        self.create_gui()
        
    def load_model(self):
        """Load the trained model and scaler with LDA transformers"""
        try:
            # Try to load the model package
            if os.path.exists('breast_cancer_model.pkl'):
                model_package = joblib.load('breast_cancer_model.pkl')
                self.final_model = model_package['model']
                self.scaler_standard = model_package['scaler']
                self.selected_features = model_package['feature_names']
                
                # Load LDA transformers and scalers if available
                self.lda_transformers = model_package.get('lda_transformers', {})
                self.lda_scalers = model_package.get('lda_scalers', {})
                self.correlated_groups = model_package.get('correlated_groups', [])
                self.lda_feature_names = model_package.get('lda_feature_names', [])
                self.single_features = model_package.get('single_features', [])
                
                self.model_loaded = True
                print("Model loaded successfully!")
                print(f"Selected features: {self.selected_features}")
                print(f"LDA transformers: {list(self.lda_transformers.keys())}")
                print(f"Single features: {self.single_features}")
            else:
                self.model_loaded = False
                print("Model file not found. Please run the training notebook first.")
        except Exception as e:
            self.model_loaded = False
            print(f"Error loading model: {e}")
    
    def apply_lda_transformation(self, input_data):
        """Apply the same LDA transformation used during training"""
        try:
            # If we don't have LDA transformers, return original data
            if not hasattr(self, 'lda_transformers') or not self.lda_transformers:
                print("No LDA transformers found, using original features")
                return input_data[self.selected_features]
            
            # Apply LDA transformation to each group
            lda_components = []
            lda_feature_names = []
            single_features_data = []
            
            # Process each correlated group
            for i, group in enumerate(self.correlated_groups, 1):
                group_name = f"Group_{i}"
                
                if len(group) == 1:
                    # Single feature - keep as is
                    if group[0] in input_data.columns:
                        single_features_data.append(input_data[group[0]].values)
                    else:
                        print(f"Warning: Single feature {group[0]} not found in input data")
                        single_features_data.append(np.zeros(len(input_data)))
                else:
                    # Multiple correlated features - apply LDA transformation
                    if group_name in self.lda_transformers and group_name in self.lda_scalers:
                        # Get group data
                        available_features = [f for f in group if f in input_data.columns]
                        if len(available_features) == len(group):
                            group_data = input_data[group]
                            
                            # Scale using the same scaler from training
                            group_scaled = self.lda_scalers[group_name].transform(group_data)
                            
                            # Apply LDA transformation
                            group_lda = self.lda_transformers[group_name].transform(group_scaled)
                            
                            # Store LDA components
                            if len(lda_components) == 0:
                                lda_components = group_lda
                            else:
                                lda_components = np.column_stack([lda_components, group_lda])
                            
                            # Add component names
                            n_components = group_lda.shape[1]
                            component_names = [f"{group_name}_LD{j+1}" for j in range(n_components)]
                            lda_feature_names.extend(component_names)
                        else:
                            print(f"Warning: Not all features available for {group_name}")
                            # Add zeros for missing LDA components
                            missing_components = np.zeros((len(input_data), 1))  # Assuming 1 component per group
                            if len(lda_components) == 0:
                                lda_components = missing_components
                            else:
                                lda_components = np.column_stack([lda_components, missing_components])
                            lda_feature_names.append(f"{group_name}_LD1")
            
            # Combine LDA components with single features
            if len(lda_components) > 0:
                # Create DataFrame with LDA components
                lda_df = pd.DataFrame(lda_components, columns=lda_feature_names, index=input_data.index)
                
                # Add single features if any
                if single_features_data:
                    single_features_array = np.column_stack(single_features_data)
                    single_feature_names = [group[0] for group in self.correlated_groups if len(group) == 1]
                    single_df = pd.DataFrame(single_features_array, columns=single_feature_names, index=input_data.index)
                    
                    # Combine LDA and single features
                    transformed_data = pd.concat([lda_df, single_df], axis=1)
                else:
                    transformed_data = lda_df
            else:
                # Only single features
                if single_features_data:
                    single_features_array = np.column_stack(single_features_data)
                    single_feature_names = [group[0] for group in self.correlated_groups if len(group) == 1]
                    transformed_data = pd.DataFrame(single_features_array, columns=single_feature_names, index=input_data.index)
                else:
                    # Fallback to original data
                    transformed_data = input_data
            
            print(f"Transformed data columns: {list(transformed_data.columns)}")
            print(f"Expected features: {self.selected_features}")
            
            # Select only the features that the model expects
            final_data = transformed_data[self.selected_features]
            return final_data
            
        except Exception as e:
            print(f"Error in LDA transformation: {e}")
            # Fallback: return original data with selected features
            return input_data[self.selected_features] if hasattr(self, 'selected_features') else input_data
    
    def predict_breast_cancer(self, new_data):
        """
        Make predictions on new breast cancer data with LDA transformation
        
        Parameters:
        new_data: pandas DataFrame with the 30 original features
        
        Returns:
        predictions: array of predictions (0=Benign, 1=Malignant)
        probabilities: array of probabilities for malignant class
        """
        print(f"Input data shape: {new_data.shape}")
        print(f"Input columns: {list(new_data.columns)}")
        
        # Apply LDA transformation to get the features the model expects
        try:
            new_data_selected = self.apply_lda_transformation(new_data)
            print(f"After LDA transformation: {new_data_selected.shape}")
            print(f"Transformed columns: {list(new_data_selected.columns)}")
        except Exception as e:
            print(f"LDA transformation failed: {e}")
            # Fallback to original approach
            new_data_selected = new_data[self.selected_features] if hasattr(self, 'selected_features') else new_data
        
        # Scale the data
        new_data_scaled = self.scaler_standard.transform(new_data_selected)
        
        # Make predictions
        predictions = self.final_model.predict(new_data_scaled)
        probabilities = self.final_model.predict_proba(new_data_scaled)[:, 1]
        
        return predictions, probabilities
    
    # ... (rest of the GUI methods remain the same)
    def create_gui(self):
        """Create the main GUI interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Breast Cancer Prediction System", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Create canvas and scrollbar for input fields
        canvas = tk.Canvas(main_frame, height=400)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
        scrollbar.grid(row=1, column=2, sticky="ns", pady=(0, 10))
        
        # Configure main_frame grid weights
        main_frame.rowconfigure(1, weight=1)
        
        # Create input fields
        self.entry_vars = {}
        self.create_input_fields(scrollable_frame)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Buttons
        predict_button = ttk.Button(button_frame, text="Predict", command=self.predict, 
                                   style='Accent.TButton')
        predict_button.grid(row=0, column=0, padx=5)
        
        clear_button = ttk.Button(button_frame, text="Clear All", command=self.clear_fields)
        clear_button.grid(row=0, column=1, padx=5)
        
        sample_button = ttk.Button(button_frame, text="Load Sample Data", command=self.load_sample_data)
        sample_button.grid(row=0, column=2, padx=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding="10")
        results_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=10)
        
        # Results text widget
        self.results_text = scrolledtext.ScrolledText(results_frame, height=8, width=80)
        self.results_text.grid(row=0, column=0, sticky="ew")
        results_frame.columnconfigure(0, weight=1)
        
        # Bind mousewheel to canvas
        self.bind_mousewheel(canvas)
        
    def create_input_fields(self, parent):
        """Create input fields for all 30 features"""
        # Group features by category
        categories = {
            'Mean Values': self.feature_names[0:10],
            'Standard Error Values': self.feature_names[10:20],
            'Worst Values': self.feature_names[20:30]
        }
        
        row = 0
        for category, features in categories.items():
            # Category header
            category_label = ttk.Label(parent, text=category, font=('Arial', 12, 'bold'))
            category_label.grid(row=row, column=0, columnspan=3, sticky="w", pady=(10, 5))
            row += 1
            
            # Features in this category
            for i, feature in enumerate(features):
                # Feature label
                label = ttk.Label(parent, text=feature.replace('_', ' ').title() + ':')
                label.grid(row=row, column=0, sticky="w", padx=(20, 5), pady=2)
                
                # Entry field
                var = tk.StringVar()
                entry = ttk.Entry(parent, textvariable=var, width=15)
                entry.grid(row=row, column=1, padx=5, pady=2)
                self.entry_vars[feature] = var
                
                # Description label
                desc_label = ttk.Label(parent, text=self.feature_descriptions[feature], 
                                     font=('Arial', 8), foreground='gray')
                desc_label.grid(row=row, column=2, sticky="w", padx=5, pady=2)
                
                row += 1
    
    def bind_mousewheel(self, canvas):
        """Bind mousewheel to canvas for scrolling"""
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def bind_to_mousewheel(event):
            canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        def unbind_from_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        canvas.bind('<Enter>', bind_to_mousewheel)
        canvas.bind('<Leave>', unbind_from_mousewheel)
    
    def validate_inputs(self):
        """Validate all input fields"""
        values = {}
        errors = []
        
        for feature_name, var in self.entry_vars.items():
            value_str = var.get().strip()
            if not value_str:
                errors.append(f"Please enter a value for {feature_name}")
                continue
            
            try:
                value = float(value_str)
                if value < 0:
                    errors.append(f"{feature_name} cannot be negative")
                else:
                    values[feature_name] = value
            except ValueError:
                errors.append(f"Invalid number format for {feature_name}")
        
        return values, errors
    
    def predict(self):
        """Make prediction based on input values"""
        if not self.model_loaded:
            messagebox.showerror("Error", "Model not loaded. Please run the training notebook first.")
            return
        
        # Validate inputs
        values, errors = self.validate_inputs()
        
        if errors:
            error_msg = "Please fix the following errors:\n\n" + "\n".join(errors[:5])
            if len(errors) > 5:
                error_msg += f"\n... and {len(errors) - 5} more errors"
            messagebox.showerror("Input Errors", error_msg)
            return
        
        try:
            # Create DataFrame with input values
            input_df = pd.DataFrame([values])
            
            # Make prediction using the same function logic
            predictions, probabilities = self.predict_breast_cancer(input_df)
            
            # Display results
            self.display_results(predictions[0], probabilities[0], values)
            
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Error making prediction: {str(e)}")
    
    def display_results(self, prediction, probability, input_values):
        """Display prediction results"""
        self.results_text.delete(1.0, tk.END)
        
        # Prediction result
        diagnosis = "MALIGNANT" if prediction == 1 else "BENIGN"
        confidence = probability if prediction == 1 else (1 - probability)
        
        result_text = f"""
PREDICTION RESULTS:
{'='*50}

DIAGNOSIS: {diagnosis}
CONFIDENCE: {confidence:.1%}
MALIGNANT PROBABILITY: {probability:.3f}
BENIGN PROBABILITY: {1-probability:.3f}

{'='*50}

INTERPRETATION:
"""
        
        if prediction == 1:
            result_text += f"""
⚠️  HIGH RISK: The model predicts this case as MALIGNANT
   - Probability of malignancy: {probability:.1%}
   - This indicates a high likelihood of cancerous cells
   - RECOMMENDATION: Immediate medical consultation required
"""
        else:
            result_text += f"""
✅ LOW RISK: The model predicts this case as BENIGN
   - Probability of malignancy: {probability:.1%}
   - This indicates a low likelihood of cancerous cells
   - RECOMMENDATION: Regular monitoring as advised by physician
"""
        
        result_text += f"""

IMPORTANT DISCLAIMERS:
• This is a machine learning prediction tool for educational purposes
• This tool should NOT replace professional medical diagnosis
• Always consult with qualified healthcare professionals
• The model has an accuracy of approximately 97% on test data
• False negatives and false positives are possible

{'='*50}

INPUT SUMMARY:
"""
        
        # Show some key input values
        key_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'concavity_mean']
        for feature in key_features:
            if feature in input_values:
                result_text += f"• {feature.replace('_', ' ').title()}: {input_values[feature]:.3f}\n"
        
        self.results_text.insert(1.0, result_text)
        
        # Color coding for the diagnosis
        if prediction == 1:
            self.results_text.tag_add("malignant", "3.11", "3.20")
            self.results_text.tag_config("malignant", foreground="red", font=("Arial", 12, "bold"))
        else:
            self.results_text.tag_add("benign", "3.11", "3.17")
            self.results_text.tag_config("benign", foreground="green", font=("Arial", 12, "bold"))
    
    def clear_fields(self):
        """Clear all input fields"""
        for var in self.entry_vars.values():
            var.set("")
        self.results_text.delete(1.0, tk.END)
    
    def load_sample_data(self):
        """Load sample data for testing"""
        # Sample benign case (approximate values)
        sample_benign = {
            'radius_mean': 12.5, 'texture_mean': 15.8, 'perimeter_mean': 82.3, 'area_mean': 477.0, 'smoothness_mean': 0.098,
            'compactness_mean': 0.107, 'concavity_mean': 0.065, 'concave points_mean': 0.025, 'symmetry_mean': 0.178, 'fractal_dimension_mean': 0.063,
            'radius_se': 0.45, 'texture_se': 1.23, 'perimeter_se': 3.2, 'area_se': 45.8, 'smoothness_se': 0.008,
            'compactness_se': 0.023, 'concavity_se': 0.018, 'concave points_se': 0.009, 'symmetry_se': 0.021, 'fractal_dimension_se': 0.004,
            'radius_worst': 14.2, 'texture_worst': 22.1, 'perimeter_worst': 95.4, 'area_worst': 615.2, 'smoothness_worst': 0.135,
            'compactness_worst': 0.215, 'concavity_worst': 0.178, 'concave points_worst': 0.082, 'symmetry_worst': 0.267, 'fractal_dimension_worst': 0.089
        }
        
        # Sample malignant case (approximate values)
        sample_malignant = {
            'radius_mean': 18.5, 'texture_mean': 22.8, 'perimeter_mean': 125.3, 'area_mean': 1105.0, 'smoothness_mean': 0.118,
            'compactness_mean': 0.198, 'concavity_mean': 0.245, 'concave points_mean': 0.125, 'symmetry_mean': 0.198, 'fractal_dimension_mean': 0.075,
            'radius_se': 0.85, 'texture_se': 1.85, 'perimeter_se': 5.8, 'area_se': 95.2, 'smoothness_se': 0.012,
            'compactness_se': 0.045, 'concavity_se': 0.055, 'concave points_se': 0.025, 'symmetry_se': 0.035, 'fractal_dimension_se': 0.008,
            'radius_worst': 24.5, 'texture_worst': 35.8, 'perimeter_worst': 165.2, 'area_worst': 1785.5, 'smoothness_worst': 0.158,
            'compactness_worst': 0.445, 'concavity_worst': 0.565, 'concave points_worst': 0.225, 'symmetry_worst': 0.325, 'fractal_dimension_worst': 0.125
        }
        
        # Ask user which sample to load
        choice = messagebox.askyesnocancel("Load Sample Data", 
                                          "Choose sample data to load:\n\n"
                                          "Yes = Benign sample\n"
                                          "No = Malignant sample\n"
                                          "Cancel = Do nothing")
        
        if choice is True:
            sample_data = sample_benign
            messagebox.showinfo("Sample Loaded", "Loaded sample BENIGN case data")
        elif choice is False:
            sample_data = sample_malignant
            messagebox.showinfo("Sample Loaded", "Loaded sample MALIGNANT case data")
        else:
            return
        
        # Load the sample data into the fields
        for feature_name, value in sample_data.items():
            if feature_name in self.entry_vars:
                self.entry_vars[feature_name].set(str(value))

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = BreastCancerPredictionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()