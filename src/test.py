class DDoSAnalysis:
    def __init__(self, file_path):
        self.raw_data = pd.read_csv(file_path)
        self.processed_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_models = {}

    def extensive_eda(self):
        print("Dataset Shape:", self.raw_data.shape)
        print("\nColumns:", list(self.raw_data.columns))
        print("\nMissing Values:\n", self.raw_data.isnull().sum())
        print("\nData Types:\n", self.raw_data.dtypes)

        numeric_cols = self.raw_data.select_dtypes(include=[np.number]).columns
        clean_numeric_data = self.raw_data[numeric_cols].replace([np.inf, -np.inf], np.nan).dropna()

        missing_values = self.raw_data.isnull().sum()
        if missing_values.any():
            missing_values = missing_values[missing_values > 0]
            plt.figure(figsize=(5, 3))
            plt.bar(missing_values.index, missing_values.values, color='purple')
            plt.title('Missing Values per Column')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            print("No Missing Values Detected.")

        if not clean_numeric_data.empty:
            num_features = clean_numeric_data.columns
            num_plots = (len(num_features) + 3) // 4

            for i in range(num_plots):
                start_index = i * 4
                end_index = min(start_index + 4, len(num_features))

                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                axes = axes.flatten()

                for j in range(4):
                    feature_index = start_index + j
                    if feature_index < len(num_features):
                        feature = num_features[feature_index]
                        axes[j].hist(clean_numeric_data[feature], bins=20, color='skyblue', edgecolor='black')
                        axes[j].set_title(f'{feature} Distribution')
                        axes[j].set_xlabel(f'{feature}')
                        axes[j].set_ylabel('Frequency')
                    else:
                        axes[j].axis('off')

                plt.tight_layout()
                plt.show()
        else:
            print("No valid numeric data for distribution plot.")

    def preprocess_data(self, target_column='Label', test_size=0.2):
        df = self.raw_data.copy()

        df = df.groupby(target_column, group_keys=False).apply(lambda x: x.sample(min(len(x), 25000), random_state=42))

        categorical_cols = df.select_dtypes(include=['object']).columns.drop(target_column, errors='ignore')
        label_encoder = LabelEncoder()
        for col in categorical_cols:
            df[col] = label_encoder.fit_transform(df[col].astype(str))

        df[target_column] = df[target_column].apply(lambda x: 0 if x.upper() == 'BENIGN' else 1)

        df['flow_efficiency'] = df['Flow Duration'] / (df['Total Fwd Packets'] + 1)
        df['burst_ratio'] = df['Flow Packets/s'] / (df['Packet Length Mean'] + 1)
        tcp_flags = ['SYN Flag Count', 'ACK Flag Count', 'PSH Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count']
        df['tcp_flag_total'] = df[tcp_flags].sum(axis=1)
        df['idle_variance'] = df['Idle Max'] - df['Idle Min']
        df['active_instability'] = df['Active Std']

        df = df.drop(['Timestamp'], axis=1, errors='ignore')

        X = df.drop(columns=[target_column])
        y = df[target_column]

        if X.isnull().any().any() or np.isinf(X.values).any():
            print("Warning: Handling NaN/Infinite values in feature set.")
            X = X.fillna(X.mean())
            X = X.replace([np.inf, -np.inf], X.max().max())
            X = X.clip(-1e6, 1e6)

        self.feature_names = X.columns
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        self.processed_data = df

    def feature_importance(self, n_features=10):
        selector = SelectKBest(score_func=f_classif, k=n_features)
        selector.fit(self.X_train, self.y_train)
        selected_features = self.feature_names[selector.get_support()]

        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(self.X_train, self.y_train)

        feature_imp = pd.DataFrame({
            'feature': selected_features,
            'importance': rf_classifier.feature_importances_[:len(selected_features)]
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_imp)
        plt.title(f'Top {n_features} Most Important Features')
        plt.tight_layout()
        plt.show()

        return feature_imp

    def train_ml_models(self):
        models = {
            'SVM': {
                'model': SVC(),
                'params': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
            },
            'Random Forest': {
                'model': RandomForestClassifier(),
                'params': {'n_estimators': [100], 'max_depth': [None, 10]}
            },
            'XGBoost': {
                'model': XGBClassifier(),
                'params': {'learning_rate': [0.01, 0.1, 0.5], 'n_estimators': [100]}
            }
        }

        results = {}
        for name, setup in models.items():
            grid_search = GridSearchCV(setup['model'], setup['params'], cv=3, scoring='f1', n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(self.X_test)

            results[name] = {
                'best_params': grid_search.best_params_,
                'model': best_model,
                'classification_report': classification_report(self.y_test, y_pred),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred),
                'accuracy': accuracy_score(self.y_test, y_pred)
            }

            self.best_models[name] = best_model

        return results

    def deep_learning_model(self):
        if len(np.unique(self.y_train)) < 2:
            print("Error: Deep learning model requires at least two classes in the training set.")
            return

        X_train = self.X_train
        X_test = self.X_test
        y_train = tf.keras.utils.to_categorical(self.y_train)
        y_test = tf.keras.utils.to_categorical(self.y_test)

        def create_model(learning_rate=0.001, units1=64, units2=32, dropout_rate=0.3):
            model = Sequential([
                Dense(units1, activation='relu', input_shape=(X_train.shape[1],)),
                BatchNormalization(),
                Dropout(dropout_rate),
                Dense(units2, activation='relu'),
                BatchNormalization(),
                Dropout(dropout_rate),
                Dense(y_train.shape[1], activation='softmax')
            ])
            model.compile(optimizer=Adam(learning_rate),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            return model

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
        ]

        model = create_model()
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )

        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()

        return {
            'classification_report': classification_report(y_test_classes, y_pred_classes),
            'confusion_matrix': confusion_matrix(y_test_classes, y_pred_classes),
            'accuracy': accuracy_score(y_test_classes, y_pred_classes)
        }
