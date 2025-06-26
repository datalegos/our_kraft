# scripts/clustering.py
# Purpose: Cluster users based on review behavior
# Author: Naveen
# Date: June 10, 2025

import pandas as pd
from pathlib import Path
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UserClustering:
    """Cluster users based on review behavior."""

    def __init__(self):
        self.n_clusters = 3  # You can adjust this as needed
        self.scaler = StandardScaler()

    def validate_input_data(self, df: pd.DataFrame) -> bool:
        """Validate that required columns exist in the DataFrame."""
        required_columns = ['user_id', 'is_toxic', 'sentiment']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            logger.error(f"Available columns: {list(df.columns)}")
            return False
        
        return True

    def engineer_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for user clustering."""
        try:
            logger.info("Engineering user features...")
            
            # Group by user and calculate features
            user_features = df.groupby('user_id').agg(
                avg_toxicity=('is_toxic', 'mean'),
                review_count=('is_toxic', 'count'),
                total_toxic_reviews=('is_toxic', 'sum'),
                negative_ratio=('sentiment', lambda x: (x == 'Negative').mean()),
                positive_ratio=('sentiment', lambda x: (x == 'Positive').mean()),
                neutral_ratio=('sentiment', lambda x: (x == 'Neutral').mean())
            ).reset_index()
            
            # Add additional derived features
            user_features['toxicity_frequency'] = (
                user_features['total_toxic_reviews'] / user_features['review_count']
            )
            
            # Calculate activity level based on review count
            user_features['activity_level'] = pd.cut(
                user_features['review_count'], 
                bins=[0, 1, 5, 10, float('inf')], 
                labels=['Low', 'Medium', 'High', 'Very High']
            )
            
            logger.info(f"Engineered features for {len(user_features)} users")
            logger.info(f"Feature columns: {user_features.columns.tolist()}")
            
            return user_features
            
        except Exception as e:
            logger.error(f"Error engineering user features: {e}")
            raise

    def cluster_users(self, df: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
        """
        Cluster users based on their review features and save the result.
        Args:
            df (pd.DataFrame): DataFrame with classified reviews.
            output_path (str | Path): Path to save clustered user data.
        Returns:
            pd.DataFrame: DataFrame with cluster labels.
        """
        try:
            logger.info("Starting user clustering...")
            
            # Validate input data
            if not self.validate_input_data(df):
                raise ValueError("Input data validation failed")
            
            # Check if we have any data
            if len(df) == 0:
                raise ValueError("Input DataFrame is empty")
            
            # Check if user_id column has valid data
            if df['user_id'].isna().all():
                raise ValueError("All user_id values are NaN")
            
            # Remove rows with NaN user_id
            df_clean = df.dropna(subset=['user_id'])
            if len(df_clean) == 0:
                raise ValueError("No valid user_id found after cleaning")
            
            logger.info(f"Processing {len(df_clean)} reviews from {df_clean['user_id'].nunique()} unique users")

            # Engineer user features
            user_features = self.engineer_user_features(df_clean)
            
            # Check if we have enough users for clustering
            if len(user_features) < self.n_clusters:
                logger.warning(f"Only {len(user_features)} users found, but {self.n_clusters} clusters requested")
                self.n_clusters = max(1, len(user_features))
                logger.info(f"Adjusted number of clusters to {self.n_clusters}")
            
            # Select features for clustering
            clustering_features = ['avg_toxicity', 'review_count', 'negative_ratio', 'toxicity_frequency']
            
            # Ensure no NaN values in clustering features
            feature_data = user_features[clustering_features].fillna(0)
            
            # Scale features for better clustering
            scaled_features = self.scaler.fit_transform(feature_data)
            
            # Perform clustering
            logger.info(f"Performing K-means clustering with {self.n_clusters} clusters...")
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            user_features['cluster'] = kmeans.fit_predict(scaled_features)

            # Add cluster interpretations
            cluster_summary = user_features.groupby('cluster').agg({
                'avg_toxicity': 'mean',
                'review_count': 'mean',
                'negative_ratio': 'mean',
                'toxicity_frequency': 'mean'
            }).round(3)
            
            logger.info("Cluster Summary:")
            logger.info(f"\n{cluster_summary}")
            
            # Add cluster labels for interpretation
            cluster_labels = []
            for i in range(self.n_clusters):
                toxicity = cluster_summary.loc[i, 'avg_toxicity']
                activity = cluster_summary.loc[i, 'review_count']
                
                if toxicity > 0.5:
                    label = "High Risk"
                elif toxicity > 0.2:
                    label = "Medium Risk"
                else:
                    label = "Low Risk"
                    
                if activity > 10:
                    label += " - High Activity"
                elif activity > 5:
                    label += " - Medium Activity"
                else:
                    label += " - Low Activity"
                    
                cluster_labels.append(label)
            
            # Map cluster labels
            user_features['cluster_label'] = user_features['cluster'].map(
                {i: cluster_labels[i] for i in range(self.n_clusters)}
            )

            # Save clustered data
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            user_features.to_csv(output_path, index=False)
            logger.info(f"Saved clustered users to {output_path}")
            
            # Log cluster distribution
            cluster_dist = user_features['cluster_label'].value_counts()
            logger.info(f"Cluster distribution:\n{cluster_dist}")

            return user_features

        except Exception as e:
            logger.error(f"Error clustering users: {e}")
            raise

    def analyze_clusters(self, clustered_users: pd.DataFrame) -> dict:
        """Analyze and provide insights about the clusters."""
        try:
            analysis = {}
            
            for cluster_id in clustered_users['cluster'].unique():
                cluster_data = clustered_users[clustered_users['cluster'] == cluster_id]
                
                analysis[f'cluster_{cluster_id}'] = {
                    'size': len(cluster_data),
                    'avg_toxicity': cluster_data['avg_toxicity'].mean(),
                    'avg_reviews': cluster_data['review_count'].mean(),
                    'avg_negative_ratio': cluster_data['negative_ratio'].mean(),
                    'high_risk_users': len(cluster_data[cluster_data['avg_toxicity'] > 0.5])
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing clusters: {e}")
            return {}

if __name__ == "__main__":
    try:
        clustering = UserClustering()
        input_file = "data/processed/classified_reviews.csv"
        output_file = "data/processed/clustered_users.csv"
        
        # Check if input file exists
        if not Path(input_file).exists():
            logger.error(f"Input file {input_file} does not exist")
        else:
            df = pd.read_csv(input_file)
            result_df = clustering.cluster_users(df, output_file)
            
            # Analyze clusters
            analysis = clustering.analyze_clusters(result_df)
            logger.info(f"Cluster analysis: {analysis}")
            
            logger.info("Clustering completed successfully")
            
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        raise