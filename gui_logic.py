"""
CSCI-1620 — GUI Logic for Intrusion Detection Model Training
(This method makes adjusting training parameters and visualizing results more user-friendly )
================================================
Input : cicids2017_cleaned.csv  (raw, 2.52M rows, 53 cols)
        X_scaled.npy            (float32, scaled features)
        y_encoded.npy           (int, encoded labels)
        feature_names.npy       (52 feature name strings)
        scaler.pkl              (fitted RobustScaler)
        label_encoder.pkl       (fitted LabelEncoder)

Output: Trained SGDClassifier model and evaluation results displayed in the GUI
        Confusion matrix and F1 score plots
"""

try:
    from PyQt6 import uic
    from PyQt6.QtWidgets import QApplication, QMainWindow, QCheckBox
    from train_sgd_inchunks import train_model
    from matplotlib.figure import Figure
    import os.path
except Exception as e:
    print(f"Error importing modules: {e}")

'''
Handleinput class is responsible for managing the user inputs from the GUI, validating them, and orchestrating the training process.
'''

class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        uic.loadUi("ModelMakerGuiV3.ui", self)  # loads all your widgets directly onto self
        # Found this method online while looking for a way to implement a progress bar
        self.file = self.user_file_input.toPlainText() or "cicids2017_cleaned.csv"
        self.chunks = self.user_chunks_input.toPlainText() or "100_000"
        self.epochs = self.user_enochs_input.toPlainText() or "2"
        self.seed = self.user_randomseed_input.toPlainText() or "42"
        self.trainer = None
        #
        self.x_scaled = "X_scaled.npy"
        self.y_encoded = "y.encoded.npy"
        self.label_encoder = "label_encoder.pkl"
        # Connect buttons to their respective functions
        self.generate_button.clicked.connect(self.validate_input)
        self.autofill_button.clicked.connect(self.autofill_inputs)
        self.clear_button.clicked.connect(self.clear_inputs)
        # Initialize progress bar
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(100)
        self.progressBar.setValue(0)
        self.progressBar.hide()  # Hide progress bar until training starts
        #
        self.log=lambda msg, **kwargs: self.event_view_window.append(str(msg)) # Logged to provide visual feedback while training
        self.trainingcompleteflag = False # <- trigger for charts plotting
        self.missing_features_flag = False # < - Raised if supporting feature files are missing
        #
        self.chartscheckbox.stateChanged.connect(self.checkbox_toggled) # Updates checkbox plot features
        
                
    '''
        Buttons to autofill or clear values in GUI are defined below
    '''

    def autofill_inputs(self) -> None:
        # Autofill with default values if auto fill button is pushed
        self.user_file_input.setPlainText(self.file)
        self.user_chunks_input.setPlainText(self.chunks)
        self.user_enochs_input.setPlainText(self.epochs)
        self.user_randomseed_input.setPlainText(self.seed)

    def clear_inputs(self) -> None:
        # Clear all input fields if clear button is pushed
        self.user_file_input.clear()
        self.user_chunks_input.clear()
        self.user_enochs_input.clear()
        self.user_randomseed_input.clear()

    def checkbox_toggled(self):
        if self.chartscheckbox.isChecked():
            self.graphics_window.show()
            if self.trainingcompleteflag == True:
                fig = self.trainer.plot_f1_bars(log=self.log)
                fig.savefig("f1_scores.png")
                self.display_plot(fig)
        else:
            self.graphics_window.hide()

    '''
        The valid input looks for the raw CSV file in the current directory
        then checks a "data" folder in the parent directory if not found in the current directory
        before giving up and asking the user to enter a valid file name.
    '''

    def validate_input(self) -> None:
        self.progressBar.show() # < - show progress bar when validation starts
        try:
            
            # The four lines below attempt convert the input values to the correct type, but will raise an error if the input is invalid (e.g. non-integer for chunks/epochs/seed or file that doesn't exist)
            self.chunks = int(self.user_chunks_input.toPlainText()) 
            self.epochs = int(self.user_enochs_input.toPlainText())
            self.seed = int(self.user_randomseed_input.toPlainText())

            # FOR LATER USE IN UPGRADED VERSION
            # ---------------------------------
            #infile = self.<user_file_input>
            # while not (os.path.isfile(infile)):
            #     self.event_view_window.append("File not found - Checking other directories...")
            #     alt_path = os.path.join("..", "data", infile)
            #     if os.path.isfile(alt_path):
            #         infile = alt_path
            #         self.event_view_window.append(f"File found: {infile}")
            #         break
            #     else:
            #         self.event_view_window.append("File does not exist in other directories\nPlease enter a valid file name.\n or add data to ~/intrusiongui/data/")
            #         return
                    
            self.event_view_window.append("Input validated successfully!") # < - reports back to gui
            self.progressBar.setValue(5) # < - updates progress bar
            QApplication.processEvents() # < - allows GUI to update before continuing with training
            self.validate_fields(self.chunks, self.epochs, self.seed)  # < - pass converted values

        except Exception as e: # Error net to catch unexpected issues in function
            import traceback
            self.event_view_window.append(f"Error occurred while validating inputs: {e}")
            self.event_view_window.append(traceback.format_exc())  # prints the full stack trace to GUI log
            return
        self.progressBar.setValue(6) # < - updates progress bar
        QApplication.processEvents() # < - allows GUI to update before continuing with training

        '''
        Validate_fields makes sure input is > 0 and random seed is non-negative before proceeding to check for the required features and training the model.
        '''

    def validate_fields(self, chunks, epochs, seed) -> None:  # receive as params

        chunks: int = chunks
        epochs: int = epochs   
        seed: int = seed

        if chunks <= 0:
            self.event_view_window.append("Chunk size must be greater than 0.")
            return
        if epochs <= 0:
            self.event_view_window.append("Epochs must be greater than 0.")
            return
        if seed < 0:
            self.event_view_window.append("Random seed must be non-negative.")
            return
        self.progressBar.setValue(7) # < - updates progress bar
        QApplication.processEvents()  # < - allows GUI to update before continuing with training
        self.event_view_window.append("All fields validated successfully!\nChecking features next...")
        self.check_features()

    '''
    The check features method logic looks for preproccessed features in the current directory
    If the required files are not found, it checks a "data" folder in the parent directory 
    before giving up and asking the user to run the preprocessing script.
    '''

    def check_features(self) -> None:
        self.progressBar.setValue(8) # < - updates progress bar
        QApplication.processEvents()  # < - allows GUI to update before continuing with training

        if (os.path.isfile('X_scaled.npy') == False):
                self.event_view_window.append("Missing preprocess feature: X_scaled.npy\nChecking Data Folder...")
                alt_path = os.path.join("..", "data", "X_scaled.npy")
                if os.path.isfile(alt_path):
                    self.x_scaled = alt_path
                    self.event_view_window.append(f"Feature found: {self.x_scaled}")  
                else:                 
                    self.event_view_window.append("Feature not found in Data Folder")
                    self.missing_features_flag = True

        if (os.path.isfile('y_encoded.npy') == False):
                self.event_view_window.append("Missing preprocess feature: y_encoded.npy\nChecking Data Folder...")
                alt_path = os.path.join("..", "data", "y_encoded.npy")
                if os.path.isfile(alt_path):
                    self.y_encoded = alt_path
                    self.event_view_window.append(f"Feature found: {self.y_encoded}")  
                else:                 
                    self.event_view_window.append("Feature not found in Data Folder")
                    self.missing_features_flag = True

        if (os.path.isfile('label_encoder.pkl') == False):
                self.event_view_window.append("Missing preprocess feature: label_encoder.pkl\nChecking Data Folder...")
                alt_path = os.path.join("..", "data", "label_encoder.pkl")
                if os.path.isfile(alt_path):
                    self.label_encoder = alt_path
                    self.event_view_window.append(f"Feature found: {self.label_encoder}")  
                else:                 
                    self.event_view_window.append("Feature not found in Data Folder")
                    self.missing_features_flag = True

        if self.missing_features_flag: 
            self.event_view_window.append("Missing required training features.\nPlease run the preprocessing script to generate the required features.")
            return
        else:
            self.progressBar.setValue(9) # < - updates progress bar
            QApplication.processEvents()  # < - allows GUI to update before continuing with training
            self.event_view_window.append("All required features found! Ready to train the model.")
            self.begin_train_model() # < - Proceeds to train the model if all features are found
        
    '''
    The show_plot function is responsible for displaying the generated F1 score charts in the GUI.
    It saves the plot to disk and then loads it into a scrollable dialog for viewing.
    '''    

    def display_plot(self, fig) -> None:
        import io
        from PyQt6.QtGui import QPixmap
        from PyQt6.QtWidgets import QGraphicsScene
        # Convert matplotlib figure to pixmap
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        
        pixmap = QPixmap()
        pixmap.loadFromData(buf.read())
        
        # Display in QGraphicsView
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        self.graphics_window.setScene(scene)
        self.graphics_window.fitInView(scene.itemsBoundingRect())
    

    '''
    The begin_train_model function initializes the training process by creating an instance of the train_model class,
    loading the preprocessed features and labels, computing class weights, and then training the model.
    '''

    def begin_train_model(self) -> None:
        self.progressBar.setValue(10) # < - updates progress bar
        QApplication.processEvents()  # < - allows GUI to update before continuing with training
        self.trainingcompleteflag = False
        results_log=lambda msg, **kwargs: self.results_window.append(str(msg)) # kwargs thrown in to allow for better flexibility and error-prevention
        
        self.progressBar.setValue(20) # < - updates progress bar
        QApplication.processEvents() # < - allows GUI to update before continuing with training
        self.event_view_window.append("Training model with the following parameters:")
        self.event_view_window.append(f"  Chunk size  : {self.chunks}")
        self.event_view_window.append(f"  Epochs      : {self.epochs}")
        self.event_view_window.append(f"  Random seed : {self.seed}")
        self.event_view_window.append("Beginning training …\n(this may take a few minutes)")
        self.progressBar.setValue(25) # < - updates progress bar
        QApplication.processEvents() # < - allows GUI to update before continuing with training
        self.trainer = train_model(self.chunks, self.seed, self.epochs, log=self.log)
        self.progressBar.setValue(50) # < - updates progress bar
        QApplication.processEvents() # < - allows GUI to update before continuing with training
        
        self.event_view_window.append("Loading labels and encoder …")
        self.trainer.load_files(self.y_encoded, self.label_encoder, log=self.log)
        self.trainer.compute_weights(log=self.log)
        self.progressBar.setValue(75) # <- updates progress bar
        QApplication.processEvents() # < - allows GUI to update before continuing with training

        self.trainer.train(self.x_scaled, log=self.log) 
        self.progressBar.setValue(99) # <- updates progress bar
        QApplication.processEvents() # < - allows GUI to update before continuing with training

        self.trainer.evaluate_results(log=results_log)
        self.progressBar.setValue(99) # <- updates progress bar
        self.progressBar.hide() # < - hide progress bar when training is complete
        QApplication.processEvents() # < - allows GUI to update before continuing with training
        
        self.trainingcompleteflag = True # <-- Shows charts if box is checked
            
        
            
