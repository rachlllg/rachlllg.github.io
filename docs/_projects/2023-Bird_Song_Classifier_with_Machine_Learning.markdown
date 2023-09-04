---
layout: project
title:  "Bird Song Classifier with Machine Learning"
year: 2023
description: Utilizing various machine learning algorithms (traditional, shallow neural networks, and deep neural networks) to classify bird species based on bird songs/calls.
ft_img: /assets/img/projects/bird_song_classifier/birds.jpeg
categories: Python Machine-Learning
---

<!-- LINKS -->
<div class='mb-5'>
<p class='mt-3 mb-3 text-center' style="font-size:0.75em;">
  <a href="#description" style='text-decoration: none;'>DESCRIPTION</a> &#183;
  <a href="#background" style='text-decoration: none;'>BACKGROUND</a> &#183;
  <a href="#motivation" style='text-decoration: none;'>MOTIVATION</a> &#183;
  <a href="#data-source" style='text-decoration: none;'>DATA SOURCE</a> &#183;
  <a href="#data-processing" style='text-decoration: none;'>DATA PROCESSING</a> &#183;
  <a href="#eda" style='text-decoration: none;'>EDA</a> &#183;
  <a href="#models" style='text-decoration: none;'>MODELS</a> &#183;
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
  <p>Notably, at the time of this project, the BirdCLEF 2023 competition had already ended, so the goal of the project was not to create a model for the competition submission, but rather to use the dataset to create a bird species classifier using machine learning techniques.</p>
  <p>Since the audio data for all 264 species cannot fit into Google Colab free version, we selected 3 species from 3 different families to build a bird song classifier for the 3 selected species only. As the test data provided by the competition contained unknown labels, we did not use the test data provided by the competition for our project. Instead, we split the training data to training, validation, and test sets for building our machine learning models.</p>
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

<!-- DATA PROCESSING -->
<div class='mb-5' id='data-processing'>
  <h3 class='mb-3'><u>DATA PROCESSING</u></h3>
  <!-- Data Preprocessing -->
  <h5 class='mb-3'><strong>1. Data Preprocessing</strong></h5>
  <p>Below is a summary and a video recording of the top level data preprocessing steps I performed, the Google Colab notebook shown in the video is the preprocessing.ipynb file in the GitHub repo.</p>
  <div class='text-center mb-3'>
    <iframe src="https://www.youtube.com/embed/cIsSjAP4Tj8?si=9IhL6LRpBJELg55l" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    <p>Data Preprocessing</p>
  </div>
  <ol>
    <li>Include only the species selected.</li>
    <p>As noted in the previous section, only 3 species (barswa, comsan, and eaywag1) were selected for this project.</p>
    <li>Remove duplicate.</li>
    <p>Instances with the same 'duration', 'type', 'location', 'primary_label', and 'author' appear to be duplicates and were removed from the dataset.</p>
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
  <!-- Data Cleaning -->
  <h5 class='mb-3'><strong>2. Data Cleaning</strong></h5>
  <p>After the top level preprocessing steps, I performed data cleaning on the train and test datasets, as summarized and shown in the video recordings below. The Google Colab notebook shown in the videos is the data_cleaning.ipynb file in the GitHub repo.</p>
  <div class='row mb-3'>
    <div class='col-md-6 text-center'>      
      <iframe src="https://www.youtube.com/embed/KBQjOAEZZSc?si=qjQ_SxYiDCYKWAXJ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
      <p class='text-center'>Data Cleaning - Part 1</p>
    </div>
    <div class='col-md-6 text-center'>
      <iframe src="https://www.youtube.com/embed/uf9nMfKFgnc?si=ccDk4JHG1rQ_PX_8" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
      <p class='text-center'>Data Cleaning - Part 2</p>
    </div>
  </div>
  <ol>
    <li>Inspect each column for NaN values.</li>
    <p>Only latitude and longitude columns contained NaN values, but only 17 out of the more than 1000 training examples had NaN latitude and longitude so I just left them as is.</p>
    <li>Inspect each column for outliers or things that would require special attention.</li>
    <p>There wasn't outliers that really stood out in this step, but instead, I noted down some columns that should be removed and some columns that could be cleaned up a bit which I performed below.</P>
    <li>Drop unused columns.</li>
    <p>'secondary_labels', 'scientific_name', 'common_name', 'author', 'license', and 'url' columns are not useful for our analysis so they were dropped from our data.</p>
    <li>Clean up the 'type' column.</li>
    <p>Some 'type' contained the bird gender and lifestage which were not related to call or song types so I summarized all types to either 'call', 'song', 'blank', or 'both'.</p>
    <li>Extract country and continent from latitude and longitude.</li>
  </ol>
  <!-- Data Extraction -->
  <h5 class='mb-3'><strong>3. Data Extraction</strong></h5>
  <p>As we already saw, the primary feature of the project is audio clips of bird song recordings. When working with the data, I discovered that it is time consuming to reload the audio clips using librosa.load() every time I want to access and use the audio objects, therefore, I used librosa.load() to load the audio files once and then extracted and save the returned NumPy array object to be used as my primary feature instead. The video below goes over the steps I performed to extract and save the NumPy array objects. The Google Colab notebook shown in the videos is the data_extraction.ipynb file in the GitHub repo.</p>
  <div class='text-center mb-3'>
    <iframe src="https://www.youtube.com/embed/se0_icqomLo?si=I5gecCw4N1XqdOcN" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    <p>Data Extraction</p>
  </div>
</div>

<!-- EDA -->
<div class='mb-5' id='eda'>
  <h3 class='mb-3'><u>EDA</u></h3>
  <p>The data is now cleaned up and ready for EDA. As I had never worked with audio data before, I looked at the notebooks from prior year BirdCLEF competitions to gauge how to work with audio data more efficiently. Listed below are some notebooks that I looked at.</p>
  <ul>
    <li><a href="https://www.kaggle.com/competitions/birdclef-2021/discussion/243463">BirdCLEF 2021 2nd place</a></li>
    <li><a href="https://www.kaggle.com/competitions/birdclef-2022/discussion/327047">BirdCLEF 2022 1st place</a></li>
    <li><a href="https://www.kaggle.com/competitions/birdclef-2023/discussion/412808">BirdCLEF 2023 1st place</a></li>
  </ul>
</div>

<!-- MODELS -->
<div class='mb-5' id='models'>
  <h3 class='mb-3'><u>MODELS</u></h3>
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


