---
layout: project
title:  "Bird Song Classifier with Machine Learning"
year: 2023
description: Utilizing various machine learning algorithms (traditional, shallow neural networks, and deep neural networks) to classify bird species based on bird songs/calls.
ft_img: /assets/img/projects/bird_song_classifier/birds.png
categories: Python Machine-Learning
---

<!-- LINKS -->
<div class='mb-5'>
<p class='mt-3 mb-3 text-center' style="font-size:0.75em;">
  <a href="#description" style='text-decoration: none;'>DESCRIPTION</a> &#183;
  <a href="#background" style='text-decoration: none;'>BACKGROUND</a> &#183;
  <a href="#motivation" style='text-decoration: none;'>MOTIVATION</a> &#183;
  <a href="#data-source" style='text-decoration: none;'>DATA SOURCE</a> &#183;
  <a href="#data-preprocessing" style='text-decoration: none;'>DATA PREPROCESSING</a> &#183;
  <a href="#eda" style='text-decoration: none;'>EDA</a> &#183;
  <a href="#model" style='text-decoration: none;'>MODELS</a> &#183;
  <a href="#limitations" style='text-decoration: none;'>LIMITATIONS</a> &#183;
  <a href="#github" style='text-decoration: none;'>GITHUB</a>
</p>
<div>
<hr class="m-0 mb-3">

<!-- DESCRIPTION -->
<div class='mb-5' id='description'>
  <h3 class='mb-3'><u>DESCRIPTION</u></h3>
  <p>This project utilized various machine learning algorithms (traditional, shallow neural networks, and deep neural networks) to classify bird species based on bird songs/calls. Data was sourced from the BirdCLEF 2023 kaggle competition, and all data cleaning, analysis, and model building were conducted using the Python programming language.</p>
</div>

<!-- BACKGROUND -->
<div class='mb-5' id='background'>
  <h3 class='mb-3'><u>BACKGROUND</u></h3>
  <p>This was the final project for the Applied Machine Learning class in my Masters in Data Science program. The original project involved four team members including myself, for the showcase here, I have only presented the work I have done, unless noted otherwise.</p>
  <p>The project was very open-ended, the teams are free to select any topic of interest and any dataset pertaining to that topic, with the objective to build a machine learning model. </p>
  <p>All work was done in Google Colab (Free) with CPU only, with Python as the programming language. Notable Python packages used:
    <ul>
      <li>standard: numpy, pandas</li>
      <li>audio processing: librosa</li>
      <li>modeling: scikit-learn, tensorflow</li>
      <li>visualization: matplotlib, seaborn</li>
    </ul>
  </p>
</div>

<!-- MOTIVATION -->
<div class='mb-5' id='motivation'>
  <h3 class='mb-3'><u>MOTIVATION</u></h3>
  <p>During the kickoff, the team proposed a number of different ideas for the project. In addition to the bird song classifier project we ultimated landed on, the team also debated working on computer vision, regression, or NLP projects. We ultimated landed on an audio classifier project for below reasons:
    <ol>
      <li>We would like to work with unstructured data.</li>
      <li>We all intended to take further classes on computer vision and NLP so we'd like to save computer vision and NLP projects to a later time.</li>
      <li>We would like to work on a project that would be better suited for deep neural networks.</li>
    </ol>
  </p>
  <p>Once the team agreed on pursuing an audio classifier project, we each searched for datasets containing audio data and agreed unanimously that a bird song classifier is both interesting and challenging enough for our project.</p>
  <p>While no team member came from a biosience background, it was interesting to learn that scientists often carry out observer-based surveys to track changes in habitat biodiversity. These surveys tend to be costly and logistics challenging, while <strong>a machine learning-based approach that can identify bird species using audio recordings</strong> would allow scientists to explore the relationship between restoration interventions and biodiversity on a larger scale, with greater precision, and at a lower cost.</p>
</div>

<!-- DATA SOURCE -->
<div class='mb-5' id='data-source'>
  <h3 class='mb-3'><u>DATA SOURCE</u></h3>
  <p>We obtained our data from the <a href='https://www.kaggle.com/competitions/birdclef-2023/overview'>BirdCLEF 2023 kaggle competition</a> hosted by the Cornell Lab of Ornithology. For the competition, the training dataset contained short recordings of individual bird calls for 264 different bird species across the globe. The audio recordings were sourced from <a href='https://xeno-canto.org/'>xenocanto.org</a> and all audio files were downsampled to 32 kHz where applicable and stored in the ogg format.</p> 
  <p> Here is an example audio clip.</p>
  <audio controls class="mb-3">
    <source src="/assets/img/projects/bird_song_classifier/XC379322.ogg" type="audio/ogg">
  Your browser does not support the audio element.
  </audio>
  <p>In addition to audio recordings of the bird calls, the training data also contained additional metadata such as secondary bird species, call type, location, auality rating, and taxonomy. The test labels were hidden for the submission purpose. Below is the first 5 rows of the training data as provided by the competition.</p>
  <pre class="csv-table">
    <table>
      <thead>
        <tr>
          <th>primary_label</th>
          <th>secondary_labels</th>
          <th>type</th>
          <th>latitude</th>
          <th>longitude</th>
          <th>scientific_name</th>
          <th>common_name</th>
          <th>author</th>
          <th>license</th>
          <th>rating</th>
          <th>url</th>
          <th>filename</th>
        </tr>
      </thead>
      <tbody>
        {% assign main = site.data.projects.bird_song_classifier.train_metadata_first_5 %}
        {% for row in (0..4) %}
        <tr>
          <td>{{ main[row].primary_label }}</td>
          <td>{{ main[row].secondary_labels }}</td>
          <td>{{ main[row].type }}</td>
          <td>{{ main[row].latitude }}</td>
          <td>{{ main[row].longitude }}</td>
          <td>{{ main[row].scientific_name }}</td>
          <td>{{ main[row].common_name }}</td>
          <td>{{ main[row].author }}</td>
          <td>{{ main[row].license }}</td>
          <td>{{ main[row].rating }}</td>
          <td>{{ main[row].url }}</td>
          <td>{{ main[row].filename }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </pre>
  <p>For our project, however, due to limitation in computing resources, we did not attempt to build a classifier for all 264 bird species. We selected 3 species from 3 different families to build a bird song classifier for the 3 selected species only. The training data was filtered to include only the 3 selected species and then was split to training, validation, and test sets for building our machine learning models. As the test data provided by the competition contained unknown labels, we did not use the test data provided by the competition for our project.</p>
  <p>The 3 species selected for the project are as follows:</p>
  <div class="container w-75" style="color: #333; background-color: #fff;">
  <div class="row">
    <div class="col-md-3 pt-3">
      <strong>Primary Label</strong>
    </div>
    <div class="col-md-3 col-sm-6 p-3 text-center" style="background-color: #ea5c4aff;">
      barswa
    </div>
    <div class="col-md-3 col-sm-6 p-3 text-center" style="background-color: #747fee;">
      comsan
    </div>
    <div class="col-md-3 col-sm-6 p-3 text-center" style="background-color: #f6a065;">
      eaywag1
    </div>
  </div>
  <div class="row">
    <div class="col-md-3 pt-3">
      <strong>Common Name</strong>
    </div>
    <div class="col-md-3 col-sm-6 p-3 text-center" style="background-color: #ea5c4aff;">
      Barn Swallow
    </div>
    <div class="col-md-3 col-sm-6 p-3 text-center" style="background-color: #747fee;">
      Common Sandpiper
    </div>
    <div class="col-md-3 col-sm-6 p-3 text-center" style="background-color: #f6a065;">
      Western Yellow Wagtail
    </div>
  </div>
  <div class="row">
    <div class="col-md-3 pt-3">
      <strong>Family</strong>
    </div>
    <div class="col-md-3 col-sm-6 p-3 text-center" style="background-color: #ea5c4aff;">
      Hirundinidae (Swallows)
    </div>
    <div class="col-md-3 col-sm-6 p-3 text-center" style="background-color: #747fee;">
      Scolopacidae (Sandpipers and Allies)
    </div>
    <div class="col-md-3 col-sm-6 p-3 text-center" style="background-color: #f6a065;">
      Motacillidae (Wagtails and Pipits)
    </div>
  </div>
  <div class="row">
    <div class="col-md-3 pt-3">
      <strong>Image</strong>
    </div>
    <div class="col-md-3 col-sm-6 p-3 text-center" style="background-color: #ea5c4aff;">
      <img class="img-fluid" src="/assets/img/projects/bird_song_classifier/barswa.jpeg" alt="barn swallow">
    </div>
    <div class="col-md-3 col-sm-6 p-3 text-center" style="background-color: #747fee;">
      <img class="img-fluid" src="/assets/img/projects/bird_song_classifier/comsan.jpeg" alt="common sandpiper">
    </div>
    <div class="col-md-3 col-sm-6 p-3 text-center" style="background-color: #f6a065;">
      <img class="img-fluid" src="/assets/img/projects/bird_song_classifier/eaywag1.jpeg" alt="western yellow wagtail">
    </div>
  </div>
</div>
</div>

<!-- DATA PREPROCESSING -->
<div class='mb-5' id='data-preprocessing'>
  <h3 class='mb-3'><u>DATA PREPROCESSING</u></h3>
  <p>Below is a summary and a video recording of the top level data preprocessing steps performed, the Google Colab notebook shown in the video can be found in the preprocessing.ipynb file in the GitHub repo.</p>
  <div class='text-center mb-3'>
    <iframe src="https://www.youtube.com/embed/cIsSjAP4Tj8?si=9IhL6LRpBJELg55l" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    <p>Preprocessing</p>
  </div>
  <ol>
    <li>Include only the species selected.</li>
    <p>As noted in the previous section, only 3 species (barswa, comsan, and eaywag1) were selected for this project.</p>
    <li>Remove duplicate.</li>
    <p>Instances with the same 'duration', 'type', 'location', 'primary_label', and 'author' appear to be duplicates and was removed from the dataset.</p>
    <li>Train/Test split.</li>
    <p>To prevent data leakage, the data was split to train and test dataset at 70/30 split.</p>
  </ol>
  <div class='row mb-3 w-75 mx-auto'>
    <div class='col-8 p-2 text-center' style="background-color: #ffab40; color: #333; border: 1px solid #333;">
      <p class='mb-0'>Train</p>
      <p class='mb-0'>70%</p>
    </div>
    <div class='col-4 p-2 text-center' style="background-color: #eeeeee; color: #333; border: 1px solid #333;">
      <p class='mb-0'>Test</p>
      <p class='mb-0'>30%</p>
    </div>
  </div>
  <p>After the top level preprocessing steps, I further performed some data cleaning on the train and test dataframes, as summarized and shown in the video recordings below. The Google Colab notebook shown in the video can be found in the data_cleaning.ipynb file in the GitHub repo.</p>
  <div class='row mb-3'>
    <div class='col-md-6 text-center'>      
      <iframe src="https://www.youtube.com/embed/KBQjOAEZZSc?si=qjQ_SxYiDCYKWAXJ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
      <p class='text-center'>Data Cleaning Part 1</p>
    </div>
    <div class='col-md-6 text-center'>
      <iframe src="https://www.youtube.com/embed/uf9nMfKFgnc?si=ccDk4JHG1rQ_PX_8" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
      <p class='text-center'>Data Cleaning Part 2</p>
    </div>
  </div>
  <ol>
    <li>Inspect each column for NaN values.</li>
    <p>Only latitude and longitude columns contains NaN values, which would result in unknown country and continent, but only 17 out of the more than 1000 training examples had NaN latitude and longitude so I just left them as is.</p>
    <li>Inspect each column for outliers or things that would require special attention.</li>
    <p></P>
    <li>Drop unused columns.</li>
    <p>'secondary_labels', 'scientific_name', 'common_name', 'author', 'license', and 'url' columns are not useful for our analysis so they were dropped from our data.</p>
    <li>Clean up the 'type' column.</li>
    <p>Some 'type' contains the bird gender and lifestage which is not particularly related to call or song types so they were summarized to a handful of types only.</p>
    <li>Extract country and continent from latitude and longitude.</li>
  </ol>
</div>

<!-- EDA -->
<div class='mb-5' id='eda'>
  <h3 class='mb-3'><u>EDA</u></h3>
</div>

<!-- MODEL -->
<div class='mb-5' id='model'>
  <h3 class='mb-3'><u>MODEL</u></h3>
</div>

<!-- LIMITATIONS -->
<div class='mb-5' id='limitations'>
  <h3 class='mb-3'><u>LIMITATIONS</u></h3>
</div>

<!-- GITHUB -->
<div class='mb-5' id='github'>
  <h3 class='mb-3'><u>GITHUB</u></h3>
  <p>Please see my <a href="https://github.com/rachlllg/Project_Bird-Song-Classifier-with-Machine-Learning">GitHub</a> for the code for the project.</p>
</div>


