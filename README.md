# EUR-FX-PCA-Clustering-Pipeline
A modular Python class for extracting and clustering latent factors from a panel of 20 EUR‐bilateral exchange rates.


- **Data Loading**  
  - Reads historical exchange rates CSV  
  - Drops any currency series with > 50% missing observations  

- **Preprocessing & Stationarity**  
  - Forward‐fills and back‐fills missing values  
  - Applies Augmented Dickey–Fuller test to identify non-stationary series  
  - First‐differences each non-stationary currency  

- **Principal Component Analysis**  
  - Standardizes the cleaned data  
  - Determines the minimal # of components needed to explain ≥ 2/3 of total variance  
  - Generates a Scree/Elbow plot for visual confirmation  

- **Clustering**  
  - Applies K-Means to the PCA scores  
  - Visualizes clusters in the first two principal components  
