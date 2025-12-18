# La Garçonne Clothing Price Prediction Model
Heather Choi, Lucy Cheng, Stephanie Pratikna

# 1. Project Overview
## 1.1 Summary
Following the presentation of our luxury market research and further discussion with our client, The Luxury Store (TLS), the company is seeking further detail regarding competitive clothing pricing. Our client has expressed a desire to price competitively against La Garçonne as  it aligns closest with TLS’s brand due to overlapping product offerings and target demographic.

## 1.2 Motivation
Our team’s goal was to create a clothing price prediction model trained using La Garçonne's products to provide competitive pricing ranges and insight into the company’s pricing strategy for our clients. 

# 2. Overview of Dataset
Our dataset consisted of 622 rows and 15 columns. The columns included features ranging from products sold, product category, and price, to more robust information, such as various product details and links to product images, as seen in **Figure 1**. Since our first project with TLS, we have since improved and expanded the La Garçonne dataset by scraping and collecting more products, and enhancing product details by adding a column for materials that identifies the primary material used to produce each item, as well as a column for product images. These changes increase the amount of data, as well as the number of informative features available for model training, theoretically improving the model’s ability to predict apparel prices.

![data-image](/assets/data_overview.png)
<sub>**Figure 1.** </sub>

# 3. Methodology
## 3.1 Data Preprocessing & Feature Engineering 
To preprocess our data for analysis, we used the Vision Transformer (ViT) model from Hugging Face to embed product images, the TfidfVectorizer to convert product descriptions and materials into numbers, OneHotEncoder to turn categorical data into binaries, and StandardScalar to standardize numerical data. 

## 3.2 Overview of Models
When building our price prediction model, we tested and compared two contending bases for it: 
1. Random Forest Model
2. XGBoost Model
   
We chose to look at Random Forest and XGBoost models because they generate higher level predictions given nonlinear predictive features, such as the ones in our dataset, and handle differing data types well with minimal preprocessing. 

In order to draw broader, more generalized performance insights and comparisons, we also established several potential baselines to compare the model’s predictions against, such as a simple linear regression and DummyRegressor models for global average price and category average price models.

## 3.3 Model Evaluation & Training
After building the Random Forest and XGBoost models, we evaluated their performance using mean squared error (MSE) and root mean squared error (RMSE). We chose RMSE as the primary metric for model selection because it reflects the error in the most straightforward way, allowing us to compare and interpret the performance of models and the prediction error directly. 

![rmse-compare](/assets/rmse_compare.png) <br/>
<sub>**Figure 2.** Comparison of RMSE across models.</sub> 

We created a bar chart (**Figure 2**) comparing the performance of our two machine learning models with the baseline models developed earlier. From **Figure 2**, it can be seen that the XGBoost model achieves the lowest RMSE, indicating the smallest average deviation between predicted apparel prices and actual market prices.
We fine tuned the performance of our XGBoost model by using GridSearch to test various depths, learning rates, and estimators, applying the best of these parameter values to the XGBoost model before training it on our dataset. 

# 4. Results and Interpretation
## 4.1 Predicted vs. Actual 

![pred-act](/assets/predicted_vs_actual_g.png)
<sub>**Figure 3.** Comparison of XGBoost predicted prices and actual product prices across categories.</sub>

A detailed graph of the performance of the XBG Model is shown in **Figure 3**, which depicts how the XGB predictions compared to the actual prices of different products across differing categories. Many of our predictions are populated around the diagonal line of perfect prediction. However, as product prices increase, predictions appear to become significantly more volatile. 

![resid-dist](/assets/residual_dist.png)
<sub>**Figure 4.** Distribution of residuals by product category.</sub>

To explore this trend further, we created a table (**Figure 4**) to analyze the model's residuals by category to understand where predictions were more or less uncertain. **Figure 4** indicates that categories with higher price variability, such as outerwear, showed larger residuals, supporting inaccuracies in our model when predicting products of generally higher price dispersions and higher prices. Therefore, it appears that there is some correlation between higher prices and heightened prediction volatility.

## 4.2 Pricing Ranges

![pricing-range](/assets/pricing_range_g.png)
<sub>**Figure 5.** Expected product pricing ranges across categories.</sub>

Using our XGBoost average price predictions and average residuals across each product category, we constructed an error bar chart (**Figure 5**) depicting the range of prices TLS can expect to price their products of differing categories to compete against industry players like and including La Garçonne . Many of the expected price ranges appear to be somewhere around the $400 to $600 mark, with several outliers including bags and jackets with ranges noticeably above $800.

## 4.3 Feature Comparison

![perm-feat](/assets/perm_feat_imp.png) <br/>
<sub>**Figure 6.** Comparison of permutation feature importance across predictive features (XGBoost).</sub>

Feature importance analysis from our XGB model, as seen in **Figure 6**, shows that product description is the strongest driver of price for La Garçonne products. This suggests that how an item is described, through keywords, plays a major role in luxury pricing, which aligns with how consumers perceive value in high-end fashion. Product category is the next most influential factor, reinforcing that different apparel types inherently command different price levels. Visual features captured through image embeddings are also highly important, highlighting the role of product presentation and imagery in pricing decisions. According to Figure 6, product materials, size guides, and its number of images appeared to contribute minimally to price differences. However, it is known that in reality, materials play a significant role in pricing. This incongruency is likely due to the repetitive and sparse nature of the material data we could extract from the La Garçonne website. 

![interpret-coefs](/assets/interpret_coefs.png) <br/>
<sub>**Figure 7.** Comparison of scaled coefficients across predictive features (simple linear regression).</sub>

Similarly, a comparison of feature importance via scaled coefficients from the linear regression baseline (**Figure 7**) also implicates description as a disproportionately strong driver of price for La Garçonne products. Furthermore, the bar chart also shows that category is the next most influential driver of price, and that the number of pictures and size guide presence are unlikely contributors. 

Overall, while **Figure 6** and **Figure 7** confirm that narrative and category positioning are key drivers of luxury apparel pricing, they also bring attention to flaws in our model.

# 5. Conclusion
After observing and comparing simple linear, DummyRegressor, Random Forest, and XGBoost predictive models, we found that the XGBoost model delivered the best performance. Following our analysis of the actual prediction performance of both the XGBoost model and its simple linear baseline model, we found that there was room to improve our predictive price model. For instance, one way we could improve our model includes expanding the dataset even further, as we found that in the process of splitting the dataset, we lost a significant amount of data that may have impacted the accuracy of our model’s predictions. We can also attempt to resolve the feature importance incongruency mentioned in **4.3** by improving our preprocessing. Instead of using the TfidfVectorizer to convert text to numbers, we can instead implement another Hugging Face language model to obtain richer text features. Another way to improve the model and further market analysis for TLS is improving generalizability by casting a wider net. If we trained our model on data from more competitors like La Garconne, predictions and derived price ranges would likely hold more utility. 

