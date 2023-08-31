---
layout: project
title:  "Evaluating the Impact of BMI on Cholesterol"
year: 2023
description: Utilizing Large Sample Ordinary Least Squares (OLS) Regression to evaluate the possible causal relationships between Body Mass Index (BMI) and cholesterol ratio.
ft_img: /assets/img/projects/bmi_cholesterol/1_in_3.png
categories: R Statistics Regression
---

<!-- LINKS -->
<div class='mb-5'>
<p class='mt-3 mb-3 text-center' style="font-size:0.75em;">
  <a href="#description" style='text-decoration: none;'>DESCRIPTION</a> &#183;
  <a href="#background" style='text-decoration: none;'>BACKGROUND</a> &#183;
  <a href="#motivation" style='text-decoration: none;'>MOTIVATION</a> &#183;
  <a href="#data-source" style='text-decoration: none;'>DATA SOURCE</a> &#183;
  <a href="#data-cleaning" style='text-decoration: none;'>DATA CLEANING</a> &#183;
  <a href="#eda" style='text-decoration: none;'>EDA</a> &#183;
  <a href="#model" style='text-decoration: none;'>MODEL</a> &#183;
  <a href="#limitations" style='text-decoration: none;'>LIMITATIONS</a> &#183;
  <a href="#github" style='text-decoration: none;'>GITHUB</a> &#183;
  <a href="#member-contribution" style='text-decoration: none;'>MEMBER CONTRIBUTIONS</a>
</p>
<div>
<hr class="m-0 mb-3">

<!-- DESCRIPTION -->
<div class='mb-5' id='description'>
  <h3 class='mb-3'><u>DESCRIPTION</u></h3>
  <p>This project utilized Large Sample Ordinary Least Squares (OLS) Regression to evaluate the possible causal relationships between Body Mass Index (BMI) and cholesterol ratio. Data was sourced from the 2005-2006 National Health and Nutrition Examination Survey (NHANES), and all data cleaning, analysis, and model building were conducted using the R programming language.</p>
</div>

<!-- BACKGROUND -->
<div class='mb-5' id='background'>
  <h3 class='mb-3'><u>BACKGROUND</u></h3>
  <p>This was the final project for the Statistics for Data Science class in my Masters in Data Science program, a collaborative effort involving me and three other classmates.</p>
  <p>The objective was to formulate a research question that distinctly identifies an independent variable (X), representing a <em>modifiable</em> 'product feature' and a dependent variable (Y), representing a 'metric of success'. The interpretation of 'product feature' and 'metric of success' was broad and extended beyond tangible products and sales.</p>
  <p>Guided by the project requirements, we utilized R as our programming language and conducted an <em>explanatory study</em> using Ordinary Least Squares (OLS) regression. While OLS regression might not be the most appropriate approach to establish causal relationships with observational data, the assignment emphasized constructing a model that is <em>reasonably plausible</em>.</p>
  <p>Notable R packages used:
    <ul>
      <li>general: tidyverse/dplyr</li>
      <li>modeling: effsize, car, lmtest, sandwich</li>
      <li>visualization: ggplot2, stargazer, knitr </li>
    </ul>
  </p>
</div>

<!-- MOTIVATION -->
<div class='mb-5' id='motivation'>
  <h3 class='mb-3'><u>MOTIVATION</u></h3>
  <div class='row mb-3'>
    <div class='col-md-5 mx-auto'>
      <img class="img-fluid p-3" src="/assets/img/projects/bmi_cholesterol/1_in_3.png" alt="1 in 3 adults in the US are overweight"> 
      <p class='text-center'>nearly 1 in 3 adults are overweight</p>
    </div>
    <div class='col-md-5 mx-auto'>
      <img class="img-fluid p-3" src="/assets/img/projects/bmi_cholesterol/2_in_5.png" alt="more than 2 in 5 adults are obese"> 
      <p class='text-center'>more than 2 in 5 adults are obese</p>
    </div>
  </div>
  <p>The increasing prevalence of obesity and its associated health complications has become a public health concern. One of the most significant health risks associated with obesity is elevated cholesterol levels, a leading risk factor for cardiovascular diseases such as heart attacks and strokes.</p>
  <p>With this motivation in mind, we proposed the research question:</p>
  <h5 class='text-center p-3'><em>Does a higher BMI cause a higher cholesterol ratio?</em></h5>
  <p class='mt-3'>where our 'product feature' is weight health, operationalized using Body Mass Index (BMI), calculated as weight in kilograms divided by height in meters squared, and our 'metric of success' is cholesterol, operationalized using cholesterol ratio, calculated as total cholesterol divided by HDL. We considered the use of total cholesterol or other cholesterol measurements but ultimately decided on the cholesterol ratio, the reasoning can be found in the final report.</p>
</div>

<!-- DATA SOURCE -->
<div class='mb-3' id='data-source'>
  <h3 class='mb-3'><u>DATA SOURCE</u></h3>
  <p>The project guideline specified that the data must be <em>cross-sectional</em> with a minimum of 200 observations, and the dependent variable must be <em>metric</em>. An example of <em>non-cross-sectional</em> or <em>longitudinal</em> data is time series, where the value from the previous day directly impacts the next day. <em>Cross-sectional</em> data is required to minimize the possible independence violation in the independent variables. An example of <em>non-metric</em> or <em>ordinal</em> data is the Likert scale, ordered categorical data, where the distance between each category is unknown.</p>
  <p>With the project guideline in mind, we obtained the data from the 2005-2006 National Health and Nutrition Examination Survey (<a href='https://doi.org/10.3886/ICPSR25504.v5'>NHANES</a>). The NHANES conducts annual surveys of individuals across the United States, and the 2005-2006 survey selected 10,348 individuals for the sample. The dataset contains separate files for demographics and results from each examination, laboratory, and questionnaire. Each row of the files represents a unique individual identifiable by a unique sequence number.</p>
  <p>With cholesterol ratio as our dependent variable and BMI as our primary independent variable, we also considered additional covariates that could impact the dependent variable. Some supplemental independent variables we included in our models are glycohemoglobin (a lab test that measures the average level of blood sugar in an individual), age, gender, and smoking habits. Due to the lack of sufficient data in our dataset, several omitted variables are excluded in our models, such as drinking habits and familial hypercholesterolemia (a genetic disorder that has a strong relationship with high cholesterol).</p>
  <p>Below is a causal diagram to visualize the various variables in the model.</p>
  <div class='text-center mb-3'>
    <img class="img-fluid p-3 mx-auto w-50 d-none d-md-block" src="/assets/img/projects/bmi_cholesterol/causal.png" alt="causal diagram">
    <img class="img-fluid p-3 w-100 d-md-none" src="/assets/img/projects/bmi_cholesterol/causal.png" alt="causal diagram">
  </div>
  <!-- Primary Variables -->
  <h5 class='mb-3'><u>Primary Variables</u></h5>
  <pre>
    <table class="table mb-0">
      <thead>
        <tr>
          <th scope="col">Dataset</th>
          <th scope="col">Variable Name</th>
          <th scope="col">Variable Description</th>
          <th scope="col">User Guide</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th scope="row">DS13 Examination: Body Measurements</th>
          <td>SEQN</td>
          <td>Respondent sequence number</td>
          <td>Page 10/129</td>
        </tr>
        <tr>
          <th scope="row"> </th>
          <td>BMXBMI</td>
          <td>Body Mass Index (kg/m**2)</td>
          <td>Page 15/129</td>
        </tr>
        <tr>
          <th scope="row">DS129 Laboratory: Total Cholesterol</th>
          <td>SEQN</td>
          <td>Respondent sequence number</td>
          <td>Page 8/65</td>
        </tr>
        <tr>
          <th scope="row"> </th>
          <td>LBXTC</td>
          <td>Total cholesterol (mg/dL)</td>
          <td>Page 8/65</td>
        </tr>
        <tr>
          <th scope="row">DS111 Laboratory: HDL Cholesterol</th>
          <td>SEQN</td>
          <td>Respondent sequence number</td>
          <td>Page 10/67</td>
        </tr>
        <tr>
          <th scope="row"> </th>
          <td>LBDHDD</td>
          <td>Direct HDL-Cholesterol (mg/dL)</td>
          <td>Page 10/67</td>
        </tr>
      </tbody>
    </table>
  </pre>
  <!-- Supplemental Variables -->
  <h5 class='mb-3'><u>Supplemental Variables</u></h5>
  <pre>
    <table class="table mb-0">
      <thead>
        <tr>
          <th scope="col">Dataset</th>
          <th scope="col">Variable Name</th>
          <th scope="col">Variable Description</th>
          <th scope="col">User Guide</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th scope="row">DS110 Laboratory: Glycohemoglobin</th>
          <td>SEQN</td>
          <td>Respondent sequence number</td>
          <td>Page 9/66</td>
        </tr>
        <tr>
          <th scope="row"> </th>
          <td>LBXGH</td>
          <td>Glycohemoglobin (%)*</td>
          <td>Page 9/66</td>
        </tr>
        <tr>
          <th scope="row">DS001 Demographics</th>
          <td>SEQN</td>
          <td>Respondent sequence number</td>
          <td>Page 8/25</td>
        </tr>
        <tr>
          <th scope="row"> </th>
          <td>RIAGENDR</td>
          <td>Gender of the sample person</td>
          <td>Page 9/25</td>
        </tr>
        <tr>
          <th scope="row"> </th>
          <td>RIDAGEYR</td>
          <td>Age in years of the sample person</td>
          <td>Page 10/25</td>
        </tr>
        <tr>
          <th scope="row">DS242 Questionnaire: Smoking - Cigarette Use</th>
          <td>SEQN</td>
          <td>Respondent sequence number</td>
          <td>Page 10/68</td>
        </tr>
        <tr>
          <th scope="row"> </th>
          <td>SMQ020</td>
          <td>Smoked at least 100 cigarettes in life</td>
          <td>Page 10/68</td>
        </tr>
        <tr>
          <th scope="row"> </th>
          <td>SMD641</td>
          <td># of days smoked cigarettes in the past 30 days</td>
          <td>Page 15/68</td>
        </tr>
        <tr>
          <th scope="row"> </th>
          <td>SMD650</td>
          <td># of cigarettes smoked/day on days that you smoked</td>
          <td>Page 16/68</td>
        </tr>
      </tbody>
    </table>
  </pre>
  <!-- Normal Range of Each Variable -->
  <h5 class='mb-3'><u>Normal Range of Each Variable</u></h5>
    <div class='row mb-0'>
        <div class='col-md-4 mx-auto'>
          <p class='mb-0 text-center'><a href='https://www.cdc.gov/obesity/basics/adult-defining.html#:~:text=If%20your%20BMI%20is%20less,falls%20within%20the%20obesity%20range'>BMI</a></p>
          <pre>
            <table class="table mb-0">
              <thead>
                <tr>
                  <th scope="col">Range</th>
                  <th scope="col">Category</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th scope="row">Below 18.5</th>
                  <td>Underweight</td>
                </tr>
                <tr>
                  <th scope="row">18.5 - 25</th>
                  <td>Healthy</td>
                </tr>
                <tr>
                  <th scope="row">25 - 30</th>
                  <td>Overweight</td>
                </tr>
                <tr>
                  <th scope="row">30 - 35</th>
                  <td>Class 1 obesity</td>
                </tr>
                <tr>
                  <th scope="row">35 - 40</th>
                  <td>Class 2 obesity</td>
                </tr>
                <tr>
                  <th scope="row">40 or above</th>
                  <td>Severe obesity</td>
                </tr>
              </tbody>
            </table>
          </pre>
        </div>
        <div class='col-md-4 mx-auto'>
          <p class='mb-0 text-center'><a href='https://www.urmc.rochester.edu/encyclopedia/content.aspx?ContentTypeID=167&ContentID=lipid_panel_hdl_ratio#:~:text=In%20general%3A,1%20is%20considered%20very%20good'>Cholesterol Ratio</a></p>
          <pre>
            <table class="table mb-0">
              <thead>
                <tr>
                  <th scope="col">Range</th>
                  <th scope="col">Category</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th scope="row">Below 3.5</th>
                  <td>Desirable</td>
                </tr>
                <tr>
                  <th scope="row">3.5 - 5</th>
                  <td>Normal</td>
                </tr>
                <tr>
                  <th scope="row">Above 5</th>
                  <td>High Risk</td>
                </tr>
              </tbody>
            </table>
          </pre>
        </div>
        <div class='col-md-4 mx-auto'>
          <p class='mb-0 text-center'><a href='https://www.cdc.gov/diabetes/basics/getting-tested.html'>Glycohemoglobin</a></p>
          <pre>
            <table class="table mb-0">
              <thead>
                <tr>
                  <th scope="col">Range</th>
                  <th scope="col">Category</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th scope="row">Below 5.7%</th>
                  <td>Normal</td>
                </tr>
                <tr>
                  <th scope="row">5.7% - 6.4%</th>
                  <td>Prediabetes</td>
                </tr>
                <tr>
                  <th scope="row">6.5% or above</th>
                  <td>Diabetes</td>
                </tr>
              </tbody>
            </table>
          </pre>
        </div>
    </div>
</div>

<!-- DATA CLEANING -->
<div class='mb-5' id='data-cleaning'>
  <h3 class='mb-3'><u>DATA CLEANING</u></h3>
  <p>Below is a summary of data cleaning steps performed, details can be found in the <a href='https://github.com/rachlllg/Evaluating-the-impact-of-BMI-on-Cholesterol/blob/main/data_cleaning.Rmd'>data_cleaning.Rmd</a> file in the GitHub repo. </p>
  <ol>
    <li>Include only respondents with valid BMI, cholesterol, and glycohemoglobin measurements.</li>
    <li>Include only respondents with valid age and gender.</li>
    <li>Calculate the cholesterol ratio by using total cholesterol divided by HDL.</li>
    <li>Merge all valid variables into one CSV, where each column represents one variable, and each row represents one observation/individual.</li>
  </ol>
  <p>After all the cleaning steps above, we identified 5,372 individuals with valid age, gender, and value measurements for BMI, Cholesterol, and Glycohemoglobin. Here are the first 5 rows of the cleaned CSV file:</p>
  <pre class="csv-table">
    <table>
      <thead>
        <tr>
          <th>SEQN</th>
          <th>BMI</th>
          <th>Cholesterol</th>
          <th>HDL</th>
          <th>Glycohemoglobin</th>
          <th>Age</th>
          <th>Gender</th>
          <th>Cholesterol_Ratio</th>
        </tr>
      </thead>
      <tbody>
        {% assign main = site.data.projects.bmi_cholesterol.main %}
        {% for row in (0..4) %}
        <tr>
          <td>{{ main[row].SEQN }}</td>
          <td>{{ main[row].BMI }}</td>
          <td>{{ main[row].Cholesterol }}</td>
          <td>{{ main[row].HDL }}</td>
          <td>{{ main[row].Glycohemoglobin }}</td>
          <td>{{ main[row].Age }}</td>
          <td>{{ main[row].Gender }}</td>
          <td>{{ main[row].Cholesterol_Ratio }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </pre>
  <p>To prevent data leakage, we selected 30% randomly from the cleaned CSV file as the exploration set with the rest set aside as the confirmation set. We built our intuition, explored the data, and built model specifications and model decisions using only the exploration set. The confirmation set was used at the end once the code to generate the regression and results were finalized.</p>
  <div class='row mb-3 w-75 mx-auto'>
    <div class='col-4 p-2 text-center' style="background-color: #ffab40; color: #333; border: 1px solid #333;">
      <p class='mb-0'>Explore</p>
      <p class='mb-0'>30%</p>
    </div>
    <div class='col-8 p-2 text-center' style="background-color: #eeeeee; color: #333; border: 1px solid #333;">
      <p class='mb-0'>Confirm</p>
      <p class='mb-0'>70%</p>
    </div>
  </div>
  <p>In addition to the independent variables shown above, we considered smoking habits as another independent variable. Unfortunately, only around 22% (1,207) of the 5,372 individuals answered they had smoked at least 100 cigarettes in their life and provided responses for the frequency of cigarette smoking. We are unable to determine the smoking habits of the other 78%. Considering the responses on the smoking frequency could be inaccurate as respondents could feel a social pressure to under-report their smoking frequency, we treated the sample of 1,207 respondents as a separate dataset from the main dataset and analyzed the effect of smoking on cholesterol ratio separately. We performed the below data cleaning steps on the smoking dataset and split the dataset to exploration and confirm sets similar to the main dataset.</p>
  <ol>
    <li>Include respondents with valid smoking habits responses only</li>
    <li>Calculate the total number of cigarettes a year by using (# of days smoked cigarettes in the past 30 days) times (# of cigarettes smoked/day on days that you smoked) </li>
  </ol>
</div>

<!-- EDA -->
<div class='mb-5' id='eda'>
  <h3 class='mb-3'><u>EDA</u></h3>
  <p>As a note, all EDAs were performed on the exploration set only. Before EDA, we removed any cases of extreme BMI above 30, extreme cholesterol ratio above 10, and extreme glycohemoglobin above 12 as they are likely not representative of the underlying overall population.</p>
  <p>Below is a summary of EDA performed, details can be found in the <a href='https://github.com/rachlllg/Evaluating-the-impact-of-BMI-on-Cholesterol/blob/main/explore.Rmd'>explore.Rmd</a> file in the GitHub repo.</p>
  <ol>
    <li>Evaluate the normality of each variable to ensure it's not heavy-tailed. Based on the histogram below, we see no strong indication of extreme heavy-tailed distribution in any variables.</li>
      <div class='row mb-3'>
        <div class='col-md-6 mx-auto'>
          <img class="img-fluid pt-3" src="/assets/img/projects/bmi_cholesterol/hist_cholesterol_ratio.png" alt="histogram of cholesterol ratio">
        </div>
        <div class='col-md-6 mx-auto'>
          <img class="img-fluid pt-3" src="/assets/img/projects/bmi_cholesterol/hist_bmi.png" alt="histogram of bmi">
        </div>
      </div>
      <div class='row mb-3'>
        <div class='col-md-6 mx-auto'>
          <img class="img-fluid pt-3" src="/assets/img/projects/bmi_cholesterol/hist_glycohemoglobin.png" alt="histogram of glycohemoglobin">
        </div>
        <div class='col-md-6 mx-auto'>
          <img class="img-fluid pt-3" src="/assets/img/projects/bmi_cholesterol/hist_age.png" alt="histogram of age">
        </div>
      </div>
    <li class='mb-3'>Evaluate the number of male and female respondents to ensure there isn't a large imbalance. Based on the histogram below, the two genders are reasonably balanced.</li>
    <div class='mb-3'>
      <img class="img-fluid mx-auto w-50 d-none d-md-block" src="/assets/img/projects/bmi_cholesterol/hist_gender.png" alt="histogram of gender">
      <img class="img-fluid mx-auto w-100 d-md-none" src="/assets/img/projects/bmi_cholesterol/hist_gender.png" alt="histogram of gender">
    </div>
    <li>Check for a correlation between each independent variable and cholesterol ratio. Based on the correlation table below, each independent variable is somewhat correlated to the cholesterol ratio, with BMI being the most correlated.</li>
    <pre>
      <table class='w-50 csv-table mx-auto'>
        <thead>
          <tr>
            <th>Independent Variables</th>
            <th>Correlation With Cholesterol Ratio</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>BMI</td>
            <td>0.3712956</td>
          </tr>
          <tr>
            <td>Glycohemoglobin</td>
            <td>0.1711966</td>
          </tr>
          <tr>
            <td>Age</td>
            <td>0.1791255</td>
          </tr>
          <tr>
            <td>Gender</td>
            <td>-0.2011416</td>
          </tr>
        </tbody>
      </table>
    </pre>
    <li class='mb-3'>Visually inspect the scatter plot of each independent variable and cholesterol ratio to identify any potential extreme clustering and evaluate the relationship between each independent variable and cholesterol ratio. The blue line on each plot indicates the general relationship trend between the two variables. Based on the scatter plots below, we can see no extreme clustering, and BMI has a positive relationship with cholesterol ratio, consistent with the correlation table above.</li>
      <div class='mb-3'>
        <img class="img-fluid mx-auto w-50 d-none d-md-block" src="/assets/img/projects/bmi_cholesterol/scatter_bmi.png" alt="scatter plot of bmi and cholesterol ratio">
        <img class="img-fluid mx-auto w-100 d-md-none" src="/assets/img/projects/bmi_cholesterol/scatter_bmi.png" alt="scatter plot of bmi and cholesterol ratio">
      </div>
      <div class='row mb-3'>
        <div class='col-md-6 mx-auto'>
          <img class="img-fluid pt-3" src="/assets/img/projects/bmi_cholesterol/scatter_glycohemoglobin.png" alt="scatter plot of glycohemoglobin and cholesterol ratio">
        </div>
        <div class='col-md-6 mx-auto'>
          <img class="img-fluid pt-3" src="/assets/img/projects/bmi_cholesterol/scatter_age.png" alt="scatter plot of age and cholesterol ratio">
        </div>
      </div>
    <li>Similar EDA steps as above were performed on the smoking dataset. There was no heavy tail distribution in the smoking dataset, but the correlation between the number of cigarettes smoked in a year and the cholesterol ratio was also relatively low, as indicated in the scatter plot below.</li>
      <div class='row mb-3'>
        <div class='col-md-6 mx-auto'>
          <img class="img-fluid pt-3" src="/assets/img/projects/bmi_cholesterol/hist_num_cigarettes.png" alt="histogram of number of cigarettes smoked in a year">
        </div>
        <div class='col-md-6 mx-auto'>
          <img class="img-fluid pt-3" src="/assets/img/projects/bmi_cholesterol/scatter_num_cigarettes.png" alt="scatter plot of number of cigarettes smoked in a year and cholesterol ratio">
        </div>
      </div>
  </ol>
</div>

<!-- MODEL -->
<div class='mb-5' id='model'>
  <h3 class='mb-3'><u>MODEL</u></h3>
  <p class='mb-0'>With cholesterol ratio as the dependent variable and BMI as the primary independent variable, considering our large sample size and the fact that there was no strong indication of a non-linear relationship between BMI and cholesterol ratio (per scatter plot), we developed Large Sample Ordinary Least Squares (OLS) Regression models of the form:</p>
  <div class='text-center p-3' style="overflow-x: auto;">
    \[
      \widehat{Cholesterol\ Ratio} = \beta_0 + \beta_1 \times BMI + Z\gamma
    \]
  </div>
  <p>where \(\beta_1\) represents the increase in cholesterol ratio for each unit increase in BMI, \(Z\) is a row vector for additional covariates, and \(\gamma\) is a column vector for coefficients.</p>
  <p>We generated multiple models by adding one covariate at a time to evaluate the effects on cholesterol ratio. The model results (as generated from the confirmation set) are in the Stargazer table below. The key coefficient on BMI was highly statistically significant in all models (as indicated by the *** next to the numbers). In the default model without any covariates, each unit of increase in BMI increased the cholesterol ratio by 0.064. Each additional covariate (glycohemoglobin, age, and gender) was also highly statistically significant in all models and contributed to improving the explanatory power of the models (as evidenced by the increase in adjusted \(R^2\)). Robust standard errors were presented in parenthesis in the models to account for any possible heteroskedastic errors.</p>
  <div class='mb-3'>
    <img class="img-fluid mx-auto w-75 d-none d-md-block" src="/assets/img/projects/bmi_cholesterol/stargazer.png" alt="stargazer table of model results">
    <img class="img-fluid mx-auto w-100 d-md-none" src="/assets/img/projects/bmi_cholesterol/stargazer.png" alt="stargazer table of model results">
  </div>
  <p>To interpret the model results, let's consider a 40-year-old female with a normal glycohemoglobin of 5.5% and a healthy BMI of 20. According to model 4, her cholesterol ratio would be 3 (20*0.058 + 5.5*0.131 + 40*0.005 + 0*0.510 + 0.924), which is highly desirable. To put this into context, a total cholesterol of 150 mg/dL with HDL of 50 mg/dL would yield a cholesterol ratio of 3. However, if her BMI were to increase to 35 (Class 1 obesity), her cholesterol ratio would increase to 3.9 (35*0.058 + 5.5*0.131 + 40*0.005 + 0*0.510 + 0.924). By age 60, assuming she had maintained the same BMI of 35 and glycohemoglobin of 5.5%, her cholesterol ratio would increase to 4 (35*0.058 + 5.5*0.131 + 60*0.005 + 0*0.510 + 0.924). A male individual of the same age with the same BMI and glycohemoglobin would have a cholesterol ratio of 4.5 (35*0.058 + 5.5*0.131 + 60*0.005 + 1*0.510 + 0.924), which would be borderline concerning. And if he is diabetic, his cholesterol ratio would be even higher.</p>
  <p>The result emphasizes the significance of maintaining a healthy BMI to lower cholesterol ratio, especially as one age. It also suggests that those with type 2 diabetes should pay special attention to their cholesterol ratio and that males should be more mindful of their cholesterol ratio than females.</p>
  <p>We also generated a similar regression model by including the number of cigarettes one smoked in a year as an additional covariate. The results are in the Stargazer table below. We can see that smoking, BMI, glycohemoglobin, and gender are statistically significant on cholesterol ratio. Among those who have smoked more than 100 cigarettes in their lifetime, holding all other factors constant, an additional 100 cigarettes smoked per year could lead to a 0.04 (100*0.0004) unit increase in cholesterol ratio. The inclusion of smoking habits further increased the explanatory power of the model, as evidenced by the increase in adjusted \(R^2\). However, we caution that the results from this model may not be generalizable to the population due to limitations in our sample, which excluded individuals who did not report their smoking history or who have never smoked. Further research with more representative samples is needed to draw definitive conclusions on the impact of smoking on cholesterol ratio.</p>
  <div class='mb-3'>
    <img class="img-fluid mx-auto w-50 d-none d-md-block" src="/assets/img/projects/bmi_cholesterol/smoking_stargazer.png" alt="stargazer table of smoking model results">
    <img class="img-fluid mx-auto w-100 d-md-none" src="/assets/img/projects/bmi_cholesterol/smoking_stargazer.png" alt="stargazer table of smoking model results">
  </div>
</div>

<!-- LIMITATIONS -->
<div class='mb-5' id='limitations'>
  <h3 class='mb-3'><u>LIMITATIONS</u></h3>
  <h5 class='mb-3'><u>Large Sample Ordinary Least Squares Regression Model Assumptions</u></h5>
  <ol>
    <strong><li>Independent and identically distributed (iid) samples</li></strong>
      <p class='mb-0'>Potential violations: </p>
      <ul>
        <li>Geographic Clustering: When collecting data, individuals were sampled from counties across the United States.</li>
        <li>Referral: Individuals may have referred friends and relatives to complete the survey.</li>
      </ul>
      <p class='mb-0'>Justification: </p>
      <ul>
        <li class='mb-3'>When collecting data, technicians deliberately oversampled individuals of certain ages and demographics to have a more accurate representation of the United States population, which may reduce some of the iid violations</li>
      </ul>
    <strong><li>A unique Best Linear Predictor exists</li></strong>
    <ul>
        <li>No Extreme heavy-tailed distributions: As can be seen in the histograms, there were no extreme heavy-tailed distributions</li>
        <li>No perfect collinearity: There was no perfect collinearity within the independent variables, as no independent variables were automatically dropped.</li>
        <li>No multicollinearity: Generally, a variance inflation factor of more than 5 would indicate high multicollinearity within the independent variables. Based on the variance inflation factor table below, we can see that no independent variables exhibited evidence of multicollinearity.</li>
        <pre>
          <table class='w-50 csv-table mx-auto'>
            <thead>
              <tr>
                <th>BMI</th>
                <th>Glycohemoglobin</th>
                <th>Age</th>
                <th>Gender</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>1.104430</td>
                <td>1.217167</td>
                <td>1.202331</td>
                <td>1.009282</td>
              </tr>
            </tbody>
          </table>
        </pre>
      </ul>
  </ol>
  <h5 class='mb-3 mt-3'><u>Omitted Variables</u></h5>
  <pre>
    <table class='table mx-auto'>
      <thead>
        <tr>
          <th>Name</th>
          <th>Description</th>
          <th>Hypothesis</th>
          <th>Bias</th>
          <th>Effects</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Familial Hypercholesterolemia</td>
          <td>A genetic disorder</td>
          <td>Individuals with familial hypercholesterolemia are known to have higher cholesterol ratios than those without</td>
          <td>Positive</td>
          <td>Driving the results away from zero and making our estimates overconfident</td>
        </tr>
        <tr>
          <td>Drinking Habits</td>
          <td>The number of alcoholic drinks an individual consumes regularly</td>
          <td>The more alcoholic drinks an individual consumes regularly, the higher their cholesterol ratios would be</td>
          <td>Positive</td>
          <td>Driving the results away from zero and making our estimates overconfident</td>
        </tr>
      </tbody>
    </table>
  </pre>
  <p>In addition to familial hypercholesterolemia and drinking habits, we also considered the possibility of diet and physical activities as omitted variables, but as an individual's diet and physical activities directly contribute to their BMI, the effect of diet and physical activities was already reflected by BMI. Therefore, we do not believe diet and physical activities are omitted variables in our analysis.</p>
  <h5 class='mb-3'><u>Reverse Causality</u></h5>
  <p>We also acknowledge the possibility of reverse causality, where an individual's cholesterol ratio could affect their glycohemoglobin levels. Although the relationship between high cholesterol and diabetes is under debate, we recognize that the positive reverse causality bias could result in overconfident estimates. We do not believe BMI is an outcome variable of any covariates in our model.</p>
  <h5 class='mb-3'><u>Explanatory Power</u></h5>
  <p>It's important to note that our adjusted \(R^2\) was below 20%, indicating there may be other factors, such as genetics and family history, with stronger effects on cholesterol ratio. Unfortunately, our sample did not contain reliable information on these variables, which limited our ability to include them in our analysis. Future researchers may consider exploring these factors in more detail.</p>
</div>

<!-- GITHUB -->
<div class='mb-5' id='github'>
  <h3 class='mb-3'><u>GITHUB</u></h3>
  <p>Please see my <a href="https://github.com/rachlllg/Project_Evaluating-the-impact-of-BMI-on-Cholesterol">GitHub</a> for the code and final report for the project.</p>
</div>

<!-- MEMBER CONTRIBUTION -->
<div class='mb-5' id='member-contribution'>
  <h3 class='mb-3'><u>MEMBER CONTRIBUTION</u></h3>
  <p>For the confidentiality of the team members, I will designate them as Member A, Member B, and Member C. As the sole team member without a full-time job during the project's execution, I undertook a substantial portion of the workload. Nevertheless, the project would not have been possible without every member of the team working together! Thank you all!</p>
  <ul>
    <li>Rachel: Find dataset, propose research question, data cleaning, EDA, model, report (data & methodology, results, limitations), slides (Model & Results)</li>
    <li>Member A: Review cleaned data & model, report (introduction & conclusion), slides (Data)</li>
    <li>Member B: Review cleaned data & model, report (limitation), slides (Limitation & Conclusion)</li>
    <li>Member C: Review cleaned data & model, slides (Introduction & Research Question)</li>
  </ul>
</div>

