---
layout: project
title:  "Bird Song Classifier with Machine Learning"
year: 2023
description: Utilizing various machine learning algorithms (traditional, shallow neural networks, and deep neural networks) to classify bird species based on bird songs/calls. Data was sourced from the BirdCLEF 2023 kaggle competition, and all data cleaning, analysis, and model building were conducted using the Python programming language.
ft_img: /assets/img/projects/bird_song_classifier/birds.jpeg
categories: Python MachineLearning Classification DSP
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
  <a href="#model-prep" style='text-decoration: none;'>MODEL PREPARATION</a> &#183;
  <a href="#training" style='text-decoration: none;'>TRAINING</a> &#183;
  <a href="#inference" style='text-decoration: none;'>INFERENCE</a> &#183;
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
  <p>All work was done in Google Colab (Free), with Python as the programming language. For training the deep neural networks (FFNN, CNN, and RNN), GPU is recommended to speed-up the processing time. Notable Python packages used:
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
  <p>Since the audio data for all 264 species are too large to fit into Google Colab free version, we reduced the scope of the task and only selected 3 species from 3 different families to build a bird song classifier for the 3 selected species. As the test data provided by the competition contained unknown labels, we did not use the test data provided by the competition for our project. Instead, we split the training data to training, validation, and test sets for building our machine learning models.</p>
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
  <h5 class='mb-3'><strong>A. Data Preprocessing</strong></h5>
  <p>Summarized below are the top level data preprocessing steps I performed, the Google Colab notebook shown in the video is the a.preprocessing.ipynb file in the 1.preprocessing folder of the GitHub repo.</p>
  <div class="row mb-3">
    <div class="col-md-6">
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
    </div>
    <div class="col-md-6 d-flex align-items-center justify-content-center">
      <iframe src="https://www.youtube.com/embed/cIsSjAP4Tj8?si=9IhL6LRpBJELg55l" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    </div>
  </div>
  <!-- Data Cleaning -->
  <h5 class='mb-3'><strong>B. Data Cleaning</strong></h5>
  <p>After the top level preprocessing steps, the data was cleaned as summarized below. The Google Colab notebook shown in the videos is the b.data_cleaning.ipynb file in the 1.preprocessing folder of the GitHub repo.</p>
  <div class="row mb-3">
    <div class="col-md-6">
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
    </div>
    <div class="col-md-6 d-flex flex-column align-items-center justify-content-center">
      <div class="mb-5 text-center">
        <iframe src="https://www.youtube.com/embed/KBQjOAEZZSc?si=qjQ_SxYiDCYKWAXJ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
        <p class='text-center'>Part 1 (Steps 1-4)</p>
      </div>
      <div class="mb-3 text-center">
        <iframe src="https://www.youtube.com/embed/uf9nMfKFgnc?si=ccDk4JHG1rQ_PX_8" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
        <p class='text-center'>Part 2 (Step 5)</p>
      </div>
    </div>
  </div>
  <!-- Data Extraction -->
  <h5 class='mb-3'><strong>C. Data Extraction</strong></h5>
  <div class="row mb-3">
    <div class="col-md-6">
      <p>One thing I discovered while working on the project was that loading the audio clips using librosa.load() is time consuming. librosa.load() takes in audio files as parameter and returns the audio object in a NumPy array. The same NumPy array object can be passed as parameters to other librosa functions to extract audio features. To save downstream processing time, I used librosa.load() to load the audio files and saved the returned NumPy array object to disk, which enabled me to use the NumPy array object directly when extracting audio features. The Google Colab notebook shown in the videos is the c.data_extraction.ipynb file in the 1.preprocessing folder of the GitHub repo.</p>
    </div>
    <div class="col-md-6 d-flex align-items-center justify-content-center">
      <iframe src="https://www.youtube.com/embed/se0_icqomLo?si=I5gecCw4N1XqdOcN" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    </div>
  </div>
</div>

<!-- EDA -->
<div class='mb-5' id='eda'>
  <h3 class='mb-3'><u>EDA</u></h3>
  <p>To get a better understanding of what audio features would be appropriate for this project, what EDA could be performed on the features, and what machine learning algorithms are most suited for audio classification tasks, I looked at the notebooks from prior year BirdCLEF competitions and read a number of articles/papers that used audio features to build machine learning model. Listed below are some notable resources that was considered when performing feature extraction, EDA, and model building for this project.</p>
  <ul class='mb-3'>
    <li><a href="https://www.kaggle.com/competitions/birdclef-2021/discussion/243463">BirdCLEF 2021 2nd place</a></li>
    <li><a href="https://www.kaggle.com/competitions/birdclef-2022/discussion/327047">BirdCLEF 2022 1st place</a></li>
    <li><a href="https://www.kaggle.com/competitions/birdclef-2023/discussion/412808">BirdCLEF 2023 1st place</a></li>
    <li><a href="https://medium.com/@LeonFedden/comparative-audio-analysis-with-wavenet-mfccs-umap-t-sne-and-pca-cb8237bfce2f">Comparative Audio Analysis With Wavenet, MFCCs, UMAP, t-SNE and PCA</a></li>
    <li><a href="https://towardsdatascience.com/cnns-for-audio-classification-6244954665ab">CNNs for Audio Classification</a></li>
    <li><a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6479959/">Environment Sound Classification Using a Two-Stream CNN Based on Decision-Level Fusion</a></li>
    <li><a href="https://towardsdatascience.com/data-augmentation-techniques-for-audio-data-in-python-15505483c63c">Data Augmentation Techniques for Audio Data in Python</a></li>
  </ul>
  <!-- General EDA -->
  <h5 class='mb-3'><strong>A. General EDA</strong></h5>
  <div class="row">
    <div class="col-md-6">
      <p>To build more generalizable models, EDAs are performed on training set only, so as to not gaining any information from the test set.</p>
      <p>Summarized below are some general EDA performed on the training set. The Google Colab notebook shown in the video is the a.EDA.ipynb file in the 2.EDA folder of the GitHub repo.</p>
    </div>
    <div class="col-md-6 mb-3 d-flex align-items-center justify-content-center">
      <iframe src="https://www.youtube.com/embed/2U3mY00rDeA?si=G7tSgU-Ld0Cnmovz" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    </div>
  </div>
  <ol class='mb-3'>
    <li>Check for class imbalance by number of samples.</li>
      <p>The three species are relatively balanced by number of samples, with barswa having slightly fewer number of samples than the other two, but the difference is not alarming.</p>
      <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/num_samples.png" alt="check for class imbalance by number of samples">
    <li>Check for class imbalance by total duration.</li>
      <p>In general, we expect features to be of the same shape when passed into machine learning models as inputs. In the case of audio inputs, each audio clip should be of the same duration. Since audio clips in our dataset are of different duration, we will need to split the audio clips to a set duration before passing into the models. Therefore, I also checked for class imbalance by total duration.</p>
      <p>At first glance, it may seem the three classes have similar total duration, however, if each audio clip is split to 5 seconds inputs without overlapping, a 10 minutes difference would result in a difference of 120 input samples. This difference would be even larger if the audio clips are split to 3 seconds clips, or if the audio clips are split with overlapping. This imbalance in total duration could result in the model favoring the species with longer duration during training. To overcome this imbalance, we can either drop some of the oversampled class (barswa), or we can stretch the audios of the undersampled classes (comsan and eaywag1) to make the three classes having similar total duration.</p>
      <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/total_duration.png" alt="check for class imbalance by total duration">
    <li>Check for total duration by call types.</li>
      <p>The call types were cleaned up as part of the data preprocessing steps mentioned above. While barswa made both 'call' and 'song' types almost equally, eaywag1 made more 'call' type than 'song', and comsan made almost exclusively only 'call' type. This might be useful information if we want to use call types as one of the input features.</p>
      <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/call_types.png" alt="check for total duration by call types">
    <li>Check for total duration by quality rating.</li>
      <p>In the original dataset, the quality rating attribute is on a scale of 0.0-5.0, presumably with 0.0 being the worst quality and 5.0 being the best quality. To better visualize the quality rating, I turned the attribute to binary, with audios of ratings above 3.0 being 'good'. All three classes had similar total durations of 'good' quality recordings, while barswa had more 'bad' quality recordings than the other two. Downsampling the 'bad' quality barswa recordings could be another way to overcome the class imbalance by total duration issue mentioned above.</p>
      <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/quality_rating.png" alt="check for total duration by quality rating">
    <li>Check for geolocation distribution by species.</li>
      <p>As part of the preprocessing, I extracted the continents of each audio sample based on the latitude and longitude. Majority of the audio clips were recorded in Europe with some in Asia and Africa. Interestingly, none of the comsan and eaywag1 audios were recorded in Americas, which might be a valuable distinguishing feature between barswa and the other two species.</p>
      <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/continent.png" alt="check for geolocation distribution by species">
  </ol>
  <!--  Audio Features -->
  <h5 class='mb-3'><strong>B. Audio Features</strong></h5>
  <p>Once the audio NumPy array objects had been extracted from the raw audio files using librosa.load(), they were passed as parameters to various librosa feature extraction functions to extract the relevant audio features. Below summarized are some of the key audio features commonly used for audio classification tasks, in particular, MFCC and melspectrograms appear to be the most useful based on existing research.</p>
  <p>Here is a visual representation of the different features derived from the 5 second audio clip below. The code used to generate this visualization can be found in the b.audio_features.ipynb file in the 2.EDA folder of the GitHub repo.</p>
  <audio controls class="mb-3">
    <source src="/assets/img/projects/bird_song_classifier/XC587730.ogg" type="audio/ogg">
    Your browser does not support the audio element.
  </audio>
  <ul class='mb-3'>
    <li class='mb-3'>Soundwave.</li>
      <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/soundwave.png" alt="soundwave">
    <li class='mb-3'>Melspectrogram & Mel-Frequency Cepstral Coefficients (MFCC): visualization of the power distribution of audio frequencies, transformed into the mel scale to better represent human perception of sound.</li>
      <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/mfcc_melspectrogram.png" alt="mfcc & mel-spectrogram">
    <li class='mb-3'>RMS energy: a measure of the signal's magnitude or "loudness" over time</li>
      <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/rms.png" alt="RMS energy">
    <li class='mb-3'>Spectral Centroid: a feature that represents the "center of mass" of the spectrum, in another word, the ‘brightness’ of the sound over time</li>
      <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/spectral_centroid.png" alt="spectral centroid">
    <li class='mb-3'>Chroma: a feature that summarizes the 12 different pitch classes</li>
      <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/chroma.png" alt="chroma">
  </ul>
  <!-- Audio Augmentation -->
  <h5 class='mb-3'><strong>C. Audio Augmentation</strong></h5>
  <p>Augmentation is an important consideration when working with audio data. Some common augmentation techniques are:</p>
  <ul>
    <li>Adding gaussian noise to the audio.</li>
    <li>Shifting the entire audio along the time axis.</li>
    <li>Changing the pitch of the audio.</li>
    <li>Stretching the entire audio along the time axis.</li>
  </ul>
  <p>Below is a visual representation of how the origianl 5 second audio soundwave changes with each augmentation technique. The code used to generate this visualization can be found in the c.augmentation.ipynb file in the 2.EDA folder of the GitHub repo.</p>
  <img class="img-fluid" src="/assets/img/projects/bird_song_classifier/augmented.png" alt="augmented vs original audio soundwave">
</div>

<!-- MODEL PREPARATION -->
<div class='mb-5' id='model-prep'>
  <h3 class='mb-3'><u>MODEL PREPARATION</u></h3>
  <!-- Train/Validation Split -->
  <h5 class='mb-3'><strong>A. Train/Validation Split</strong></h5>
  <p>Since the test dataset is reserved for final inference purpose, I further split the training dataset to train and validation sets for hyperparameter tuning during training. The code used to split the training dataset can be found in the a.train_val_split.ipynb file in the 3.model_prep folder of the GitHub repo.</p>
  <div class='row mx-auto'>
    <div class='col-8 p-2 text-center' style="background-color: #ffab40; color: #333; border: 1px solid #333;">
      <p class='mb-0'>Train</p>
      <p class='mb-0'>70%</p>
    </div>
    <div class='col-4 p-2 text-center' style="background-color: #eeeeee; color: #333; border: 1px solid #333;">
      <p class='mb-0'>Test</p>
      <p class='mb-0'>30%</p>
    </div>
  </div>
  <div class='row mb-3 mx-auto'>
    <div class='col-6 p-2 text-center' style="background-color: #ffd9a8; color: #333; border: 1px solid #333;">
      <p class='mb-0'>Train</p>
      <p class='mb-0'>70%</p>
    </div>
    <div class='col-2 p-2 text-center' style="background-color: #fcead2; color: #333; border: 1px solid #333;">
      <p class='mb-0'>Val</p>
      <p class='mb-0'>30%</p>
    </div>
  </div>
  <p>To overcome the class imbalance in total duration, I downsampled the oversampled species in training set and put the rest in validation set. By doing this, the training set is now balanced, which would allow the models to learn features from the three classes equally well during training.</p>
  <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/train_duration.png" alt="total duration by species in training set">
  <p>The total duration in the validation set is imbalanced after splitting to train/val sets, as it should be representative of the potential imbalance in the test set.</p>
  <img class="img-fluid mb-5" src="/assets/img/projects/bird_song_classifier/val_duration.png" alt="total duration by species in validation set">
  <!-- Create Class Methods -->
  <h5 class='mb-3'><strong>B. Create Class Methods</strong></h5>
      <p>Even though only 3 species were selected for the project, the audio features for the training and validation set are still too large to fit into memory of Google Colab (the free version only provides 12.7GB RAM). I therefore created two class methods which allowed me to manage the memory usage more efficiently.</p>
  <div class="row">
    <div class="col-md-6">
      <p class='mb-3'>The Framed class is used to frame the audios (split audios of varying length to set lengths clips with or without augmentation, and with or without overlapping). The Extraction class is used to extract the audio features and labels from each of the framed clips (with or without normalization and/or average pooling) in a shape that's ready to be passed into the models. The code used to create and test the class methods can be found in the b.class_methods.ipynb file in the 3.model_prep folder of the GitHub repo.</p>
    </div>
    <div class="col-md-6 mb-3 d-flex flex-column align-items-center justify-content-center">
      <iframe src="https://www.youtube.com/embed/44BFY5bFt10?si=jVPH70Kzb0EfFWgp" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
      <p class='text-center mt-2'>Part 1 (Train/Val split & class methods)</p>
    </div>
  </div>
  <!-- Extract Framed Audios -->
  <h5 class='mb-3'><strong>C. Extract Framed Audios</strong></h5>
    <p>Since I intend to experiment various machine learning algorithms, it would be inefficient to frame the audios and extract the features everytime in each model notebook. Therefore, I used the Framed class (discussed above) to extract the framed audios with below specifications and saved the updated dataframe (including the 'framed' column) to disk for future use.</p>
    <ul>
      <li>5.0 seconds frame with 2.5 seconds overlap - with and without augmentation</li>
      <li>8.0 seconds frame with 4.0 seconds overlap - with and without augmentation</li>
    </ul>
    <p>Usually, pandas dataframes can be saved to disk in csv format to be reloaded as dataframes when needed, however, since the framed audios are framed using the tf.signal.frame method which returns the framed audios as an array of Tensor objects, saving arrays of Tensor objects to csv format would render the objects unusuable (or at least very difficult to parse). So, in order to save the updated dataframe in a reloadable format, the dataframes were saved to disk using the pickle library in pkl format. The code used to extract the framed audios and save the updated dataframes can be found in the c.extract_framed_audios folder in the 3.model_prep folder of the GitHub repo.</p>
  <!-- Extract Features & Labels -->
  <h5 class='mb-3'><strong>D. Extract Features & Labels</strong></h5>
  <p>Once the framed audios have been extracted, I then used the Extraction class (discussed above) to extract the various features with below specifications (all numeric features were normalized), and then saved the extracted features to disk (using pickle) for future use. The numbers in parenthesis indicate the number of each feature extracted from each audio. The code used to extract and save the features can be found in the d.extract_features_labels folder in the 3.model_prep folder of the GitHub repo.</p>
  <div class="row">
    <div class="col-md-6">
      <ul>
        <li>MFCC(20) with and without average pooling</li>
        <li>Chroma(12) with and without average pooling</li>
        <li>RMS(1) with and without average pooling</li>
        <li>Spectral Centroid(1) with and without average pooling</li>
        <li>Melspectrogram(20) with and without average pooling</li>
        <li>Continent</li>
        <li>Type</li>
        <li>Rating</li>
      </ul>
    </div>
    <div class="col-md-6 d-flex flex-column align-items-center justify-content-center">
      <iframe src="https://www.youtube.com/embed/4rdYVgKYAE8?si=H3-T70HWETzls1gg" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
      <p class='text-center mt-2'>Part 2 (Extract framed audio & features)</p>
    </div>
  </div>
  <p>To better understand the dimension of the audio features, let's assume we have one audio sample of 8 seconds in duration, sampled at 16000 sample_rate. The audio input signal would then have shape [n,] where n = (duration_in_seconds * sample_rate) = (8 * 16000) = 128000. The number of frames (n_frame) can then be calculated as (n / hop_length + 1) = (128000 / 512 + 1) = 251, where hop_length is default to window_length // 4 = 2048 // 4 = 512, where window_length is default to 2048 in librosa. If we then extracted 20 MFCCs at each time step, the resulting MFCC feature dimension would be [n_frame, n_mfcc] = [251, 20]. It's worth noting that the features returned from librosa feature extraction functions takes on dimension of [n_feature, n_frame] by default, I transposed the features to take on dimension of [n_frame, n_feature] instead so average pools and convolutions are applied along the time axis.</p>
  <p>When average pooling is applied, each audio feature of shape [n_frame, n_feature] are average pooled along the time axis, resulting in audio feature of shape [,n_feature]. In our example, the 8 seconds audio sample with 20 MFCCs would produce an average pooled feature of shape [,20]. We can therefore view average pooling as a dimensionality reduction technique, where a 2D feature of shape [251, 20] is reduce to 1D of shape [,20] by taking the average of the 251 frames for each MFCC. This process can be visualized as below.</p>
  <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/avgpool.png" alt="visualization of average pooling process">
  <p>When average pooling is not applied, each audio feature of shape [n_frame, n_feature] keeps its original dimension. If we have 100 8-seconds samples, each with 20 MFCCs, our inputs with average pooling would then have dimension [n_samples, n_features] = [100, 20], our inputs without average pooling would then have dimension [n_samples, n_frame, n_features] = [100, 251, 20]. If we were to extract and concatenate more than one feature, let's say 20 MFCCs and 12 Chroma, our 100 sample inputs would then have dimension [100, 32] with average pooling, or [100, 251, 32] without average pooling, where 32 = 20 MFCCs + 12 Chroma.</p>
</div>

<!-- TRAINING -->
<div class='mb-5' id='training'>
  <h3 class='mb-3'><u>TRAINING</u></h3>
  <p class='mt-3 mb-3 text-center' style="font-size:0.8em;">
    <a href="#baseline" style='text-decoration: none;'>Baseline</a> &#183;
    <a href="#rf" style='text-decoration: none;'>Random Forest</a> &#183;
    <a href="#xgboost" style='text-decoration: none;'>XGBoost</a> &#183;
    <a href="#svm" style='text-decoration: none;'>SVM</a> &#183;
    <a href="#logistic" style='text-decoration: none;'>Logistic Regression</a> &#183;
    <a href="#ffnn" style='text-decoration: none;'>FFNN</a> &#183;
    <a href="#1d" style='text-decoration: none;'>1D CNN</a> &#183;
    <a href="#2d" style='text-decoration: none;'>2D CNN</a> &#183;
    <a href="#lstm" style='text-decoration: none;'>LSTM RNN</a> &#183;
    <a href="#gru" style='text-decoration: none;'>GRU RNN</a> &#183;
    <a href="#transformer" style='text-decoration: none;'>Transformer</a> &#183;
    <a href="#tl" style='text-decoration: none;'>Transfer Learning</a>
  </p>
  <!-- A1. Baseline -->
  <h5 class='mb-3' id='baseline'><strong>A1. Baseline</strong></h5>
  <p>Each species represent 1/3 of the total duration in the training set, if one were to randomly guess the species, we would expect a 33% accuracy. This will serve as our baseline.</p>
  <p class='mb-3'>All notebooks for the models can be found in the 4.training folder in the GitHub repo.</p>
  <!-- B1. Ensemble - Random Forest -->
  <h5 class='mb-3' id='rf'><strong>B1. Ensemble - Random Forest</strong></h5>
  <div class="row">
    <div class="col-md-6">
      <p>Ensemble is a machine learning technique that combined multiple models to one model. Random Forest, rooted from decision tree, is one of the ensemble techniques, where multiple decision trees are used to find the most optimal predictive feature at each 'branch'. I used the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html">RandomForestClassifier</a> from sklearn to implement the random forest models.</p>
    </div>
    <div class="col-md-6 mb-3 d-flex align-items-center justify-content-center">
      <iframe src="https://www.youtube.com/embed/dhUUCJntQrg?si=gME3Xs15kPjPb3MU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    </div>
  </div>
  <p>As mentioned earlier, many different audio features (such as MFCC, RMS, etc) could be extracted and used as features in machine learning models, so I implemented random forest models with 16 different combinations of various features to find the features with the strongest predictive power. I also experimented with different framed audio duration (3 seconds with 1 second overlap, 5 seconds with 2.5 seconds overlap, and 8 seconds with 4 seconds overlap) in case the audio frame duration made a difference in the models.</p>
  <p>Summarized below are the feature combinations from the models with the highest validation accuracy for each framed audio duration, with no augmentation. All models were generated with 50 estimators, with entropy as the criterion, and with combinations of normalized and average pooled (along time axis) 20 MFCC, 20 melspectrogram, 12 chroma, 1 RMS, 1 spectral Centroid, and/or 5 one-hot encoded continents.</p>
  <pre class='csv-table'>
    <table>
      <thead>
        <tr>
          <th scope="col">Framed Duration (secs)</th>
          <th scope="col">Overlap Duration (secs)</th>
          <th scope="col">Features</th>
          <th scope="col">Training Accuracy</th>
          <th scope="col">Validation Accuracy</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>3.0</td>
          <td>1.0</td>
          <td>MFCC(20) + Spectral_Centoid(1) + Continents(5)</td>
          <td>100%</td>
          <td>69%</td>
        </tr>
        <tr>
          <td>3.0</td>
          <td>1.0</td>
          <td>MFCC(20) + RMS(1) + Continents(5)</td>
          <td>100%</td>
          <td>69%</td>
        </tr>
        <tr>
          <td>5.0</td>
          <td>2.5</td>
          <td>MFCC(20) + Continents(5)</td>
          <td>100%</td>
          <td>69%</td>
        </tr>
        <tr>
          <td>5.0</td>
          <td>2.5</td>
          <td>MFCC(20) + Chroma(12) + Continents(5)</td>
          <td>100%</td>
          <td>70%</td>
        </tr>
        <tr>
          <td>8.0</td>
          <td>4.0</td>
          <td>MFCC(20) + Spectral_Centoid(1) + Continents(5)</td>
          <td>100%</td>
          <td>71%</td>
        </tr>
        <tr style="background-color: #99FF99;">
          <td>8.0</td>
          <td>4.0</td>
          <td>MFCC(20) + RMS(1) + Continents(5)</td>
          <td>100%</td>
          <td>73%</td>
        </tr>
      </tbody>
    </table>
  </pre>
  <p>From this summary, we can see MFCC is predominently a better predictor than melspectrogram, and MFCC + continents are the strongest predictor among all combinations. The predictive power of RMS, chroma, and spectral centroid is not conclusive from this exercise alone, and even though framed audios with 8 seconds in length had the highest validation accuracy, the different to that of 5 seconds is not glaring so I would not conclude framed audios with 8 seconds in length are better than those with 5 seconds in length (at least not just yet). Notabily, framing audios to 3 seconds created more training samples, but did not improve the results. Going forward, I will only work with framing duration of 5 seconds or 8 seconds.</p>
  <p>I also tried applying random augmentation to the training samples for the 5 seconds and 8 seconds framed samples, but it did not improve the model performance. This is not to say augmentation is not useful, but rather the augmentation technique may need to be revisited. I also tried increasing the number of melspectrogram to 128 (instead of 20) which did not make a difference in model performance.</p>
  <p class='mb-3'>The best performing model had 73% validation accuracy, which is much higher than our baseline of 33% (random guess), the algorithm is performing better than I expected but consistent with the general characteristics of random forest, all models are severely overfitted to the training data. For hypertuning the models, I changed the number of estimators to 40 and 80. Neither change made notable difference in the model performance. I did not hypertune the criterion as my research suggested entropy is generally the better criterion to use.</p>
  <!-- B2. Ensemble - XGBoost -->
  <h5 class='mb-3' id='xgboost'><strong>B2. Ensemble - XGBoost</strong></h5>
  <p><a href="https://xgboost.readthedocs.io/en/stable/index.html#">XGBoost</a> is another ensemble machine learning technique that utilizes the gradient boosting framework to provide parallel tree boosting in decision tree algorithms. I used the Framed and Extraction classes for framing the audios and extracting the features in the random forest notebooks, but starting from XGBoost, I will be directly using the features saved in .pkl format to improve efficiency.</p>
  <p>Similar to the random forest models, I experimented with different combinations of features from 5 seconds framed audios and 8 seconds framed audios, with and without augmentation. Summarized below are the feature combinations from the models with the highest validation accuracy for each framed audio duration. All models were generated with 100 estimators, with dart as the booster.</p>
  <pre class='csv-table'>
    <table>
      <thead>
        <tr>
          <th scope="col">Framed Duration (secs)</th>
          <th scope="col">Overlap Duration (secs)</th>
          <th scope="col">Features</th>
          <th scope="col">Augment</th>
          <th scope="col">Training Accuracy</th>
          <th scope="col">Validation Accuracy</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>5.0</td>
          <td>2.5</td>
          <td>MFCC(20) + Chroma(12) + Continents(5)</td>
          <td>No</td>
          <td>100%</td>
          <td>70%</td>
        </tr>
        <tr>
          <td>5.0</td>
          <td>2.5</td>
          <td>MFCC(20) + Chroma(12) + Continents(5)</td>
          <td>Yes</td>
          <td>100%</td>
          <td>69%</td>
        </tr>
        <tr style="background-color: #99FF99;">
          <td>8.0</td>
          <td>4.0</td>
          <td>MFCC(20) + RMS(1) + Continents(5)</td>
          <td>No</td>
          <td>100%</td>
          <td>72%</td>
        </tr>
        <tr>
          <td>8.0</td>
          <td>4.0</td>
          <td>MFCC(20) + RMS(1) + Continents(5)</td>
          <td>Yes</td>
          <td>100%</td>
          <td>72%</td>
        </tr>
      </tbody>
    </table>
  </pre>
  <p>Similar to the findings from the random forest models, MFCC is predominently a better predictor than melspectrogram, and MFCC + chroma or MFCC + RMS together with continents are consistently the better predictor than any other feature combinations. The 8 seconds framed audios again had higher validation accuracy than those with 5 seconds, and the random augmentation applied to the audios did not play a role in improving the models. The models had similar performance as random forest and are still severely overfitted to the training data.</p>
  <div class="row">
    <div class="col-md-6">
      <p class='mb-3'>For hypertuning the models, I used <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV">GridSearchCV</a> from sklearn. The GridSearch was ran on the model with the same specifications as the highlighted model above. I also ran another model with the same specifications but replaced RMS with chroma. The GridSearch identified the best max depth at 6 with 200 estimators. However, the validation accuracy with this optimal hyperparameter setting did not improve over the original model, this is expected as the model is overfitted to the training data and already learned 100% of the features in the training data in the original model (with fewer estimators). To improve the model, more training data is likely needed.</p>
    </div>
    <div class="col-md-6 mb-3 d-flex flex-column align-items-center justify-content-center">
      <iframe src="https://www.youtube.com/embed/XOpfbJHPTOg?si=O7wyz7TcvxeYjXkh" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    </div>
  </div>
  <!-- C1. Support Vector Machine -->
  <h5 class='mb-3' id='svm'><strong>C1. Support Vector Machine (SVM)</strong></h5>
  <p>Another tranditional machine learning algorithm commonly used for classification tasks is support vector machine (SVM), which is an algorithm used to identify a hyperplane that segregates/classifies the data points in an N-dimensional space. I used <a href='https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html'>SVC</a> from sklearn to implement the SVM models.</p>
  <p>I again experimented with different combinations of features from 5 seconds framed audios and 8 seconds framed audios, with and without augmentation. Summarized below are the feature combinations from the models with the highest validation accuracy for each framed audio duration. All models were generated with C=4 (regularization parameter), with rbf as the kernel.</p>
  <pre class='csv-table'>
    <table>
      <thead>
        <tr>
          <th scope="col">Framed Duration (secs)</th>
          <th scope="col">Overlap Duration (secs)</th>
          <th scope="col">Features</th>
          <th scope="col">Augment</th>
          <th scope="col">Training Accuracy</th>
          <th scope="col">Validation Accuracy</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>5.0</td>
          <td>2.5</td>
          <td>MFCC(20) + Chroma(12) + Continents(5)</td>
          <td>No</td>
          <td>90%</td>
          <td>72%</td>
        </tr>
        <tr>
          <td>5.0</td>
          <td>2.5</td>
          <td>MFCC(20) + Chroma(12) + Continents(5)</td>
          <td>Yes</td>
          <td>87%</td>
          <td>69%</td>
        </tr>
        <tr style="background-color: #99FF99;">
          <td>8.0</td>
          <td>4.0</td>
          <td>All Features</td>
          <td>No</td>
          <td>92%</td>
          <td>74%</td>
        </tr>
        <tr>
          <td>8.0</td>
          <td>4.0</td>
          <td>MFCC(20) + Chroma(12) + Continents(5)</td>
          <td>Yes</td>
          <td>87%</td>
          <td>72%</td>
        </tr>
      </tbody>
    </table>
  </pre>
  <p class='mb-3'>The results are still consistent with the findings from the previous two models so I will omit detailed discussion here. Notabily SVM was much faster to run than XGBoost and the training results are less overfitted with comparable validation accuracy. For hypertuning the models, I again used GridSearchCV from sklearn. The hypertuning did not improve the performance of the models.</p>
  <!-- D1. Logistic Regression -->
  <h5 class='mb-3' id='logistic'><strong>D1. Logistic Regression</strong></h5>
  <div class="row">
    <div class="col-md-6">
      <p>Logistic regression is perhaps the most basic machine learning algorithm for classification tasks. Similar to the earlier models, I experimented with different combinations of features from 5 seconds framed audios and 8 seconds framed audios, with and without augmentation. Summarized below are the feature combinations from the models with the highest validation accuracy for each framed audio duration. All models used Adam optimizer, 0.005 learning rate, batch size of 32, and ran for 100 epochs.</p>
    </div>
    <div class="col-md-6 mb-3 d-flex flex-column align-items-center justify-content-center">
      <iframe src="https://www.youtube.com/embed/rT3tpKXxj3Y?si=59Z0vLndKZcIADg8" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    </div>
  </div>
  <pre class='csv-table'>
    <table>
      <thead>
        <tr>
          <th scope="col">Framed Duration (secs)</th>
          <th scope="col">Overlap Duration (secs)</th>
          <th scope="col">Features</th>
          <th scope="col">Augment</th>
          <th scope="col">Training Accuracy</th>
          <th scope="col">Validation Accuracy</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>5.0</td>
          <td>2.5</td>
          <td>MFCC(20) + Spectral Centroid(1) + Continents(5)</td>
          <td>No</td>
          <td>70%</td>
          <td>65%</td>
        </tr>
        <tr>
          <td>5.0</td>
          <td>2.5</td>
          <td>MFCC(20) + RMS(1) + Continents(5)</td>
          <td>Yes</td>
          <td>68%</td>
          <td>60%</td>
        </tr>
        <tr style="background-color: #99FF99;">
          <td>8.0</td>
          <td>4.0</td>
          <td>MFCC(20) + Spectral Centroid(1) + Continents(5)</td>
          <td>No</td>
          <td>72%</td>
          <td>65%</td>
        </tr>
        <tr>
          <td>8.0</td>
          <td>4.0</td>
          <td>MFCC(20) + RMS(1) + Continents(5)</td>
          <td>Yes</td>
          <td>71%</td>
          <td>65%</td>
        </tr>
      </tbody>
    </table>
  </pre>
  <p>As expected, the logistic regression models performed poorly (still better than baseline but worse than any of the previous traditional machine learning algorithms), they are less overfitted (which is good), but the validation accuracy are consistently lower than prior models. Logistic regression is generally better suited for simpler classification tasks, where the data is linearly separable, which is rarely the case for most real life datasets. But nevertheless, I wanted to give it a try so I can compare the performance between shallow neural network (logistic regression) and deep neural networks (FFNN, CNN, etc).</p>
  <p>To visualize the learning progress for the best performing logistic regression model (highlighted above), I plotted the below loss and accuracy progression curves for training and validation over the 100 epochs. We can see the model is making steady learning progress one epoch after another. The validation curve is pretty far apart from the training curve, indicating signs of overfitting. I also observed the zig zag pattern in the learning curves, more prominently in the validation curves, indicating the performance is unsteady, a sign that the model is struggling with the validation data one epoch after another. I should expect the two learning curves (train and validation) be closer together, with less zig zag, with a more complex architecture.</p>
  <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/logistic_regression.png" alt="logistic regression learning curves">
  <p class='mb-3'>I did not bother hypertuning the models as I know logistic regression is not the best suited algorithm for the data.</p>
  <!-- E1. Feed Forward Neural Network (FFNN) -->
  <h5 class='mb-3' id='ffnn'><strong>E1. Feed Forward Neural Network (FFNN)</strong></h5>
  <p>The first deep neural network I tried is Feed Forward Neural Network (FFNN), it is essentially a logistic regression model but with hidden layers added. The hidden layers (actived with some non-linear activation function) allow the model to find non-linear relationships between the features and the labels.</p>
  <div class="row">
    <div class="col-md-6">
      <p>Based on observations from the models above, it appears framed audios with 8 seconds in duration without augmentation have consistently outperformed others, so I will be using 8 seconds framed audios going forward. This is not to say other durations are inferior, it's just for this dataset (and with the way I processed the data), framing audios to 8 seconds seems to work better. Similarly, models with MFCC as the primary audio feature have consistently outperformed those with melspectrogram, so I will also be using only MFCC as the primary audio feature going forward. Again, many other audio models have found success in using melspectrogram and I have no doubt it is equally good or even better than MFCC, it's just I've had better performance with MFCC for this project so far.</p>
    </div>
    <div class="col-md-6 mb-3 d-flex flex-column align-items-center justify-content-center">
      <iframe src="https://www.youtube.com/embed/OoLZd-mXjA0?si=cbeh1QDvE65SUFOB" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    </div>
  </div>
  <p>Summarized below are the results with different feature combinations. All models used Adam optimizer, 0.0001 learning rate, batch size of 32, ran for 100 epochs, with 3 hidden layers, each of 128, 64, and 32 nodes respectively and activated with the ReLU activation function.</p>
  <pre class='csv-table'>
    <table>
      <thead>
        <tr>
          <th scope="col">Features</th>
          <th scope="col">Training Accuracy</th>
          <th scope="col">Validation Accuracy</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>MFCC(20) + RMS(1) + Spectral Centroid(1)</td>
          <td>81%</td>
          <td>66%</td>
        </tr>
        <tr>
          <td>MFCC(20) + RMS(1) + Continents(5)</td>
          <td>79%</td>
          <td>69%</td>
        </tr>
        <tr>
          <td>MFCC(20) + Spectral Centroid(1) + Continents(5)</td>
          <td>81%</td>
          <td>71%</td>
        </tr>
        <tr style="background-color: #99FF99;">
          <td>MFCC(20) + Chroma(12) + Continents(5)</td>
          <td>84%</td>
          <td>75%</td>
        </tr>
      </tbody>
    </table>
  </pre>
  <p>Compared to logistic regression and all prior models, FFNN had the highest validation accuracy and was the least overfitted. Below is the learning curves from the best performing model (highlighted above). Compared to the learning curves from logistic regression above, we still observe the zig zag in the validation curves, suggesting the model is still struggling with the validation data from one epoch to another.</p>
  <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/ffnn.png" alt="feed forward neural network learning curves">
  <p>To better understand how each hyperparameter changes the model performance, I did an ablation study on the best performing model (highlighted above) by changing the learning rate, batch size, number of epochs, the number of hidden layers, and the number of nodes in each hidden layer. Summarized below is the results from the ablation study, the study is not exhaustive but based on the study, the default hyperparameter settings (Adam optimizer, 0.0001 learning rate, batch size of 32, ran for 100 epochs, with 3 hidden layers, each of 128, 64, and 32 nodes) performed the best with this dataset. The validation accuracy changed slightly from the original model (highlighted above) due to randomness in initial weight initialization and shuffling.</p>
  <pre class='csv-table mb-3'>
    <table>
      <thead>
        <tr>
          <th scope="col">Hidden Layer</th>
          <th scope="col">Num Epochs</th>
          <th scope="col">Batch Size</th>
          <th scope="col">Learning Rate</th>
          <th scope="col">Training Accuracy</th>
          <th scope="col">Validation Accuracy</th>
        </tr>
      </thead>
      <tbody>
        <tr style="background-color: #99FF99;">
          <td>[128,64,32]</td>
          <td>100</td>
          <td>32</td>
          <td>0.0001</td>
          <td>0.84</td>
          <td>0.77</td>
        </tr>
        <tr>
          <td style="color: red;">[32]</td>
          <td>100</td>
          <td>32</td>
          <td>0.0001</td>
          <td>0.67</td>
          <td>0.56</td>
        </tr>
        <tr>
          <td style="color: red;">[64,32]</td>
          <td>100</td>
          <td>32</td>
          <td>0.0001</td>
          <td>0.77</td>
          <td>0.66</td>
        </tr>
        <tr>
          <td style="color: red;">[256,128,64]</td>
          <td>100</td>
          <td>32</td>
          <td>0.0001</td>
          <td>0.91</td>
          <td>0.74</td>
        </tr>
        <tr>
          <td>[128,64,32]</td>
          <td style="color: red;">200</td>
          <td>32</td>
          <td>0.0001</td>
          <td>0.90</td>
          <td>0.73</td>
        </tr>
        <tr>
          <td>[128,64,32]</td>
          <td>100</td>
          <td style="color: red;">8</td>
          <td>0.0001</td>
          <td>0.88</td>
          <td>0.71</td>
        </tr>
        <tr>
          <td>[128,64,32]</td>
          <td>100</td>
          <td style="color: red;">64</td>
          <td>0.0001</td>
          <td>0.82</td>
          <td>0.71</td>
        </tr>
        <tr>
          <td>[128,64,32]</td>
          <td>100</td>
          <td>32</td>
          <td style="color: red;">0.01</td>
          <td>0.88</td>
          <td>0.70</td>
        </tr>
        <tr>
          <td>[128,64,32]</td>
          <td>100</td>
          <td>32</td>
          <td style="color: red;">0.0005</td>
          <td>0.92</td>
          <td>0.68</td>
        </tr>
      </tbody>
    </table>
  </pre>
  <!-- F1. 1D Convolutional Neural Networks (1D CNN) -->
  <h5 class='mb-3' id='1d'><strong>F1. 1D Convolutional Neural Networks (1D CNN)</strong></h5>
  <div class="row">
    <div class="col-md-9">
      <p>Similar to FFNN models implemented above, I implemented the 1D CNN models with tensorflow functional API architecture, with 8 seconds framed audios, MFCC as the main audio feature, and without augmentation.</p>
      <p>Different from FFNN (and any other models implemented above), the audio features are no longer average pooled, but instead kept in the original 2D dimension, to be convoluted along the time axis. To the right is an animated illustration of 1D convolute along the time axis for an 8-second audio sample (at 16000 sample rate) with 20 MFCC and 12 chroma features, the features are concetenated and the yellow box represents one filter. </p>
    </div>
    <div class="col-md-3 d-flex flex-column align-items-center justify-content-center">
      <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/1DCNN.gif" alt="visualization of 1D convolute process">
    </div>
  </div>
  <div class="row">
    <div class="col-md-6 d-flex flex-column align-items-center justify-content-center">
      <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/1DCNN_summary.png" alt="1D CNN model summary">
    </div>
    <div class="col-md-6">
      <p>Also different from previous models, instead of one hot encoding the continents, I decided to create embedding representations of the continents. Embeddings are used for natural language processing tasks as learned representations of words or tokens. Since continents are words, I thought why not give embeddings a try.</p>
      <p>I used tensorflow keras Embedding() layer to create embeddings with output dimension 2 for the 6 tokens, each token represents one continent (there are 5 continents in our training data, including one 'unknown' continent), plus an additional token reserved for out-of-dictionary word. To make the embedding dimensions match the audio features dimensions, I tiled the embeddings along the time axis to change embeddings of shape [,embedding_dim] = [,2] to shape [n_frame,embedding_dim] = [251,2], effectively the embeddings for each continent for the respective audio sample is repeated along the time axis. Once the embeddings were tiled to the same shape as the audio features, the audio features and embeddings were concatenated, resulting in input features of shape [251, 34] in the example of 20 MFCC + 12 Chroma + 2 Embeddings.</p>
      <p>All models were ran with the same architecture: two 1D conv layers (each with kernel_size=5, strides=1, activated with the ReLU activation function, and L2 regularization=0.02, with the first layer having 32 filters and the second layer having 64 filters), each followed by a max pooling layer (each with pool_size=2), followed by a flattening layer and then a fully connected layer (with units=1024), before being passed to a 50% dropout layer, which finally leads to the output layer. In addition, I also utilized the 'rating' feature to create sample weights to give audio samples with worse ratings less weights during training. L2 regularization and dropout were employed to reduce overfitting, and callback technique was used to call the model back to the epoch with the highest weighted validation accuracy. All models were trained with the Adam optimizer, 0.001 learning rate, and batch size of 32. To the left is a visualization of the model architecture (with 20 MFCC and 12 Chroma + continents embeddings as features).</p>
    </div>
  </div>
  <p>Summarized below are the results with different feature combinations, utilizing the same architecture and hyper-parameters mentioned above. Compared to all prior models, there is a clear jump in performance, from 75% highest validation accuracy (FFNN) to 91% highest validation accuracy, and the models are noticeabily less overfitted than all prior models.</p>
  <pre class='csv-table'>
    <table>
      <thead>
        <tr>
          <th scope="col">Features</th>
          <th scope="col">Training Accuracy</th>
          <th scope="col">Validation Accuracy</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>MFCC(20) + Chroma(12) + Continents(2)</td>
          <td>98%</td>
          <td>87%</td>
        </tr>
        <tr>
          <td>MFCC(20) + Chroma(12) + RMS(1) + Spectral Centroid(1) + Continents(2)</td>
          <td>97%</td>
          <td>87%</td>
        </tr>
        <tr>
          <td>MFCC(20) + Spectral Centroid(1) + Continents(2)</td>
          <td>98%</td>
          <td>90%</td>
        </tr>
        <tr>
          <td>MFCC(20) + RMS(1) + Continents(2)</td>
          <td>98%</td>
          <td>91%</td>
        </tr>
        <tr style="background-color: #99FF99;">
          <td>MFCC(20) + RMS(1) + Spectral Centroid(1) + Continents(2)</td>
          <td>97%</td>
          <td>91%</td>
        </tr>
      </tbody>
    </table>
  </pre>
  <p>Below is the learning curves from the best performing model (highlighted above). It's worth noting that the learning curves are not exactly apple-to-apple comparison to all prior models, since I utilized callback technique when training the 1D CNN models, so the progression is only up until the best epoch (epoch 35) here.</p>
  <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/1DCNN_progression.png" alt="1D CNN learning curves">
  <p>Now that the model is finally performing decently well, it's important to review the confusion matrix and classification reports for the training and validation sets to get a better understanding of which species the model struggles with.</p>
  <div class="row">
    <div class="col-md-6">
      <p>To interpret the confusion matrix, let's take barswa as example. From the validation confusion matrix, we can see that when the true label is barswa, the model mistook barswa for comsan in 67 instances and mistook barswa for eaywag1 in 25 instances. In our EDA performed earlier, we know that barswa had proportionally more poor quality recordings in the training set than the other two species, but the performance on barswa is actually comparable to the other two,  this can be seen from the f1 score in the classification report as well. Notabily comsan had lower precision and eaywag1 had lower recall, where as barswa has balanced precision and recall, but the overall f1 score among all three species were comparable.</p>
    </div>
    <div class="col-md-6 d-flex flex-column align-items-center justify-content-center">
      <iframe src="https://www.youtube.com/embed/YDNtZlLNBdM?si=Nqbs_gUDu3LX3_if" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    </div>
  </div>
  <div class="row">
    <div class="col-md-6 d-flex flex-column align-items-center justify-content-center">
      <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/1DCNN_train_report.png" alt="1D CNN train classification report">
    </div>
    <div class="col-md-6 d-flex flex-column align-items-center justify-content-center">
      <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/1DCNN_val_report.png" alt="1D CNN validation classification report">
    </div>
  </div>
  <div class="row">
    <div class="col-md-6 d-flex flex-column align-items-center justify-content-center">
      <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/1DCNN_train_cm.png" alt="1D CNN train confusion matrix">
    </div>
    <div class="col-md-6 d-flex flex-column align-items-center justify-content-center">
      <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/1DCNN_val_cm.png" alt="1D CNN validation confusion matrix">
    </div>
  </div>
  <p>I further ran comparable models by omitting the continents to evaluate whether the continents contribute to the overall model performance. With continents omitted, the best performing 1D CNN model (with similar architecture as above, with the same hyper-parameters) had the highest validation accuracy of 89%, evidence that including continents as feature in our models does improve the model performance.</p>
  <div class="row mb-3">
    <div class="col-md-6">
      <p>To perform hyperparameter tuning on the best performing model (highlighted above), I utilized <a href='http://hyperopt.github.io/hyperopt/'>HyperOpt</a>, a Python library for hyperparameter optimization. Hyperparameters selected for hyperparameter tuning are: learning rate, number of hidden layers, filter size, kernel size, stride size, pooling size, regularization strength, dropout rate, number of nodes in the last fully connected layer, and batch size. The different hyperarameters did not make a notable difference in the model performance, details can be seen in the c.1DCNN_8sec_w_continents_hyperopt.ipynb file.</p>
    </div>
    <div class="col-md-6 d-flex flex-column align-items-center justify-content-center">
      <iframe src="https://www.youtube.com/embed/_JG7Aa8thgE?si=az_qSSPaVsmKfhBE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    </div>
  </div>
  <!-- F2. 2D Convolutional Neural Networks (2D CNN) -->
  <h5 class='mb-3' id='2d'><strong>F2. 2D Convolutional Neural Networks (2D CNN)</strong></h5>
  <div class="row">
    <div class="col-md-6">
      <p>Since the 1D CNN model (and prior models) showed 8 seconds framed audios without augmentation with MFCC(20) + RMS(1) + Spectral Centroid(1) + Continents(2) as features had the best performance, I implemented the 2D CNN model with tensorflow functional API architecture with these features.</p>
    </div>
    <div class="col-md-6 d-flex flex-column align-items-center justify-content-center">
      <iframe src="https://www.youtube.com/embed/fmBY6HNCwuk?si=5FfSopQVYZaoyD3-" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    </div>
  </div>
  <p>Compare to 1D CNN models where the filters only move down, the filters in 2D CNN models move in 2 directions: to the right and down. There are two ways of constructing the features for the 2D CNN model:</p>
  <ol>
    <li>use the same features as what was used in the 1D CNN model by concatenating the features so we have features of shape [n_frame, n_features], or</li>
    <li>consider each feature as a different channel, effectively creating features of shape [n_frame, n_features, n_channels]</li>
  </ol>
  <p>I decided to go with the second option, below is an illustration of the features using the second option.</p>
  <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/2DCNN.png" alt="2D CNN features">
  <p>With MFCC as the main feature of shape [n_frames, n_features] = [251, 20], RMS is tiled/repeated 20 times at each time step to match the shape of MFCC, the same was done for Spectral Centroid. Since continents embeddings had shape of [,embedding_dim] = [,2], the continents embeddings were first tiled/repeated 10 times along the features dimension and then repeated 251 times along the time dimension to match the shape of MFCC. After all features have shape [251, 20], they are stacked on top of each other to create a combined 3D features with shape [n_frame, n_features, n_channels] = [251, 20, 4], where the n_channels can be viewed similar to the number of colors in an image classification 2D CNN task.</p>
  <div class='text-center'>
  <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/2DCNN_summary.png" alt="2D CNN architecture">
  </div>
  <p>As can be seen in the model summary above, the model consisted of three 2D conv layers (each with kernel_size=(5,5), strides=(1,1), activated with the ReLU activation function, and L2 regularization=0.02, with the first layer having 64 filters and the two following layers having 32 filters each), each layer is followed by a max pooling layer (each with pool_size=(2,2)). The model is then followed by a flattening layer and then a fully connected layer (with units=1024), before being passed to a 50% dropout layer, which finally leads to the output layer. Similar to the 1D CNN model, I also utilized the 'rating' feature to create sample weights to give audio samples with worse ratings less weights during training. L2 regularization and dropout were employed to reduce overfitting, and callback technique was used to call the model back to the epoch with the highest weighted validation accuracy. The model was trained with the Adam optimizer, 0.001 learning rate, and batch size of 32.</p>
  <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/2DCNN_progression.png" alt="2D CNN learning curves">
  <p>The model reached its highest validation accuracy of 86% at epoch 31, this validation accuracy is slightly lower than that of the best performing 1D CNN model above, but still outperforms any other previously implemented models. The model is also more overfitted than the 1D CNN model above, suggesting stronger regularization or more dropout between the hidden layers might be beneficial. Notably 2D CNN is much more computationally expensive compared to 1D but the performance was not any better, this could be attributed to the fact that the model was not hypertuned and that 2D CNNs are generally better for image classification tasks when we have an audio classification task. The classification reports and confusion matrixes are presented below for completeness sake, but I won't discuss them in detail.</p>
  <div class="row">
    <div class="col-md-6 d-flex flex-column align-items-center justify-content-center">
      <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/2DCNN_train_report.png" alt="2D CNN train classification report">
    </div>
    <div class="col-md-6 d-flex flex-column align-items-center justify-content-center">
      <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/2DCNN_val_report.png" alt="2D CNN validation classification report">
    </div>
  </div>
  <div class="row">
    <div class="col-md-6 d-flex flex-column align-items-center justify-content-center">
      <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/2DCNN_train_cm.png" alt="2D CNN train confusion matrix">
    </div>
    <div class="col-md-6 d-flex flex-column align-items-center justify-content-center">
      <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/2DCNN_val_cm.png" alt="2D CNN validation confusion matrix">
    </div>
  </div>
  <!-- G1. Recurrent Neural Networks - Long Short-Term Memory (LSTM RNN) -->
  <h5 class='mt-3 mb-3' id='lstm'><strong>G1. Recurrent Neural Networks - Long short-term memory (LSTM RNN)</strong></h5>
  <div class="row">
    <div class="col-md-6">
      <p>Long Short-Term Memory (LSTM) is one of the Recurrent Neural Networks (RNN), developed to solve the vanishing gradient and memoery loss issue with the traditional RNNs. Like RNNs in generally, LSTM are commonly used for sequential data, such as text and audios.</p>
      <p>For the LSTM models, I used 8 seconds framed audios without augmentation with audio features only. Similar to what I had done with the previous models, I experimented with different combinations of audio features using the same model architecture and hyperparameters and found that the combination of 20 MFCC + 1 Spectral Centroid had the best performance, as evidenced in the table below.</p>
    </div>
    <div class="col-md-6 d-flex flex-column align-items-center justify-content-center">
      <iframe src="https://www.youtube.com/embed/xiKCFVnKwGU?si=TNctHlQvv-Y0c2QN" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    </div>
  </div>
  <pre class='csv-table mb-3'>
    <table>
      <thead>
        <tr>
          <th scope="col">Features</th>
          <th scope="col">Training Accuracy</th>
          <th scope="col">Validation Accuracy</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>MFCC(20) + Chroma(12)</td>
          <td>0.98</td>
          <td>0.86</td>
        </tr>
        <tr>
          <td>MFCC(20) + RMS(1) + Spectral Centroid(1)</td>
          <td>0.99</td>
          <td>0.89</td>
        </tr>
        <tr>
          <td>MFCC(20) + Chroma(12) + RMS(1) + Spectral Centroid(1)</td>
          <td>0.99</td>
          <td>0.90</td>
        </tr>
        <tr>
          <td>MFCC(20) + RMS(1)</td>
          <td>0.98</td>
          <td>0.91</td>
        </tr>
        <tr style="background-color: #99FF99;">
          <td>MFCC(20) + Spectral Centroid(1)</td>
          <td>0.97</td>
          <td>0.91</td>
        </tr>
      </tbody>
    </table>
  </pre>
  <div class="row">
    <div class="col-md-2 d-flex flex-column align-items-center justify-content-center">
      <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/LSTM_summary.png" alt="LSTM model summary">
    </div>
    <div class="col-md-10">
      <p>All models were ran with the same architecture: one LSTM layer with 256 nodes, followed by a BatchNormalization layer and a dropout layer with dropout=0.3, followed by three dense layers (each with 256, 128, and 48 nodes respectively, activated with the ReLU activation function, and L2 regularization=0.01) each followed by a dropout layer (dropout=0.3), before being passed to the output layer. Differ from the FFNN and CNNs where the Keras Functional API architecture was used, I used the Keras Sequential modeling for the LSTMs, adhering to the sequential nature of the LSTM model architecture. The 'rating' feature was used to create sample weights to give audio samples with worse ratings less weights during training. L2 regularization and dropout were employed to reduce overfitting, and callback technique was used to call the model back to the epoch with the highest weighted validation accuracy. All models were trained with the Adam optimizer, 0.002 learning rate, and batch size of 64. Note that another teammember experimented on the number of dense layers and the nodes for each layer and the specifications used here is based on the optimal results from that experiment.</p>
      <p>A visualization of the model architecture can be seen on the left, this visualization uses 20 MFCC + 12 Chroma as the audio features for each 8 seconds framed audio.</p>
      <p>Below is the learning curves from the best performing model (highlighted above). Compared to the learning curves from the CNN models, we observe more prominent zig zags, suggesting the model is still struggling from one epoch to another.</p>
    </div>
  </div>
  <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/LSTM_progression.png" alt="LSTM learning curves">
  <p>By comparing the confusion matrix and classification reports for the training and validation sets against the best performing 1D CNN model above, we see that the performance between the two models are comparable, with the LSTM model being ever so slightly less overfitted and showing marginally better recall for barswa and comsan than the 1D CNN model.</p>
  <div class="row">
    <div class="col-md-6 d-flex flex-column align-items-center justify-content-center">
      <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/LSTM_train_report.png" alt="LSTM train classification report">
    </div>
    <div class="col-md-6 d-flex flex-column align-items-center justify-content-center">
      <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/LSTM_val_report.png" alt="LSTM validation classification report">
    </div>
  </div>
  <div class="row">
    <div class="col-md-6 d-flex flex-column align-items-center justify-content-center">
      <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/LSTM_train_cm.png" alt="LSTM train confusion matrix">
    </div>
    <div class="col-md-6 d-flex flex-column align-items-center justify-content-center">
      <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/LSTM_val_cm.png" alt="LSTM validation confusion matrix">
    </div>
  </div>
  <!-- G2. Recurrent Neural Networks - Gated Recurrent Unit (GRU RNN) -->
  <h5 class='mt-3 mb-3' id='gru'><strong>G2. Recurrent Neural Networks - Gated Recurrent Unit (GRU RNN)</strong></h5>
  <p>Similar to LSTM, Gated Recurrent Unit (GRU) is also a Recurrent Neural Network (RNN). Differ from LSTM, it employs a gated mechanism and uses fewer parameters. The GRU models were ran using the same features, model archiecture, and hyperparameters as the LSTM models, with the only difference being the LSTM layer was replaced with the GRU layer. Below is a summary of the model results.</p>
  <pre class='csv-table mb-3'>
    <table>
      <thead>
        <tr>
          <th scope="col">Features</th>
          <th scope="col">Training Accuracy</th>
          <th scope="col">Validation Accuracy</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>MFCC(20) + Spectral Centroid(1)</td>
          <td>0.98</td>
          <td>0.91</td>
        </tr>
        <tr>
          <td>MFCC(20) + Chroma(12) + RMS(1) + Spectral Centroid(1)</td>
          <td>1.00</td>
          <td>0.92</td>
        </tr>
        <tr>
          <td>MFCC(20) + Chroma(12)</td>
          <td>1.00</td>
          <td>0.93</td>
        </tr>
        <tr>
          <td>MFCC(20) + RMS(1) + Spectral Centroid(1)</td>
          <td>0.99</td>
          <td>0.93</td>
        </tr>
        <tr style="background-color: #99FF99;">
          <td>MFCC(20) + RMS(1)</td>
          <td>0.99</td>
          <td>0.93</td>
        </tr>
      </tbody>
    </table>
  </pre>
  <p>The model with 20 MFCC + 1 RMS + 1 Spectral Centroid performed equally well as the model without Spectral Centroid, suggesting the addition of Spectral Centroid did not add substantial value to the model performance, therefore, I will focus on the model without Spectral Centroid for the analysis of the results. This is the best performing model so far, with the validation accuracy at 93%, exceeding the validation accuracy of all previous models.</p>
  <p>Compared to the learning curves from the LSTM model, the zig zags are less predominent. And the two learning curves are more closely together with the training curve performing slightly better than the validation curve, suggesting less overfitting and overall better performance than all previous models.</p>
  <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/GRU_progression.png" alt="GRU learning curves">
  <p>By comparing the confusion matrix and classification reports for the training and validation sets against the best performing 1D CNN model above, we see that the GRU model outperforms the 1D CNN model in almost all metrics for all three species.</p>
  <div class="row">
    <div class="col-md-6 d-flex flex-column align-items-center justify-content-center">
      <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/GRU_train_report.png" alt="GRU train classification report">
    </div>
    <div class="col-md-6 d-flex flex-column align-items-center justify-content-center">
      <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/GRU_val_report.png" alt="GRU validation classification report">
    </div>
  </div>
  <div class="row">
    <div class="col-md-6 d-flex flex-column align-items-center justify-content-center">
      <img class="img-fluid" src="/assets/img/projects/bird_song_classifier/GRU_train_cm.png" alt="GRU train confusion matrix">
    </div>
    <div class="col-md-6 d-flex flex-column align-items-center justify-content-center">
      <img class="img-fluid" src="/assets/img/projects/bird_song_classifier/GRU_val_cm.png" alt="GRU validation confusion matrix">
    </div>
  </div>

  <!-- H1. Transformer -->
  <h5 class='mt-3 mb-3' id='transformer'><strong>H1. Transformer</strong></h5>
  <p>I did not experiment with using a transformer architecture at the time of the project, a <a href='/project/2024-Bird_Song_Classifier_with_Vision_Transformer/'>separate project</a> is dedicated to experiments conducted using a vision transformer architecture on the same dataset.</p>

  <!-- J1. Transfer Learning -->
  <h5 class='mt-3' id='tl'><strong>J1. Transfer Learning</strong></h5>
  <p>A <a href='/project/2024-Bird_Song_Classifier_with_Vision_Transformer/'>separate project</a> is dedicated to experiments conducted using transfer learning on the same dataset.</p>
</div>

<!-- INFERENCE -->
<div class='mb-5' id='inference'>
  <h3 class='mb-3'><u>INFERENCE</u></h3>
  <div class="row">
    <div class="col-md-6">
      <p>Now that the trainings are completed, it is finally time to run inference on the test data. Given the 1D CNN and the GRU RNN models performed the best amongst all models, I ran inference using the trained weights from these two models.</p>
      <p class='mb-3'>All notebooks for the inference can be found in the 5.inference folder in the GitHub repo.</p>
    </div>
    <div class="col-md-6 d-flex flex-column align-items-center justify-content-center">
      <iframe src="https://www.youtube.com/embed/9fyq2-wE4KI?si=-B1ZY3OMstZAkOXi" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    </div>
  </div>
  <!-- A1. Data Preparation -->
  <h5 class='mb-3'><strong>A1. Data Preparation</strong></h5>
  <p>The inference should be run using the same features as was used for training. First, the framed audios were extracted using the a1.inference_8sec.ipynb notebook, where the Framed class method is used. This is the same notebook in the 3.model_prep/c.extract_framed_audios/8sec.ipynb notebook that was used on the training data, note that I changed the class method slightly to accommodate for the test data.</p>
  <p>Once the framed audios were extracted, the labels and audio features were extracted using the a2.inference_8_sec_audio_features_not_avgpooled.ipynb notebook by passing in the train_df and test_df to the Extraction class method. This is the same notebook as 3.model_prep/d.extract_feature_labels/8_sec_audio_features_not_avgpooled.ipynb notebook. It is important to pass in both train_df and test_df as the feature normalization scaler should be fit using the training data to prevent data leakage.</p>
  <p>Similarly, the non-audio features were extracted using the a3.inference_8_sec_non_audio_features.ipynb notebook by passing in the train_df and test_df to the Extraction class method. This is the same notebook as 3.model_prep/d.extract_feature_labels/8_sec_non_audio_features.ipynb notebook. It is not neccessary to pass in train_df but I still passed in the train_df for convenience sake (so I wouldn't need to make adjustments to the Extraction class).</p>
  
  <!-- B1. 1D CNN -->
  <h5 class='mt-3 mb-3'><strong>B1. 1D CNN</strong></h5>
  <p>It is common to combine the training and validation set as one combined training set to train the best performing model with the same hypertuned hyperparameters one last time before using the final model for inference, so that is what I did for running the inference as well. I combined the training and validation set to one training dataset, used 20 MFCC, 1 RMS, 1 Spectral Centroid, with continents of embedding size = 2 as the main features with rating as sample weights, using the same hyperparameters as the hypertuned training model to train the model one last time. The model was trained for 35 epochs with no callback. Once the model is done training, I ran inference on the test set using the same features. Below summarized is the resulting classification report and the confusion matrix.</p>
  <div class="row">
    <div class="col-md-6 d-flex flex-column align-items-center justify-content-center">
      <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/1DCNN_inference_train_report.png" alt="1D CNN inference train classification report">
    </div>
    <div class="col-md-6 d-flex flex-column align-items-center justify-content-center">
      <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/1DCNN_inference_test_report.png" alt="1D CNN inference test classification report">
    </div>
  </div>
  <div class="row">
    <div class="col-md-6 d-flex flex-column align-items-center justify-content-center">
      <img class="img-fluid" src="/assets/img/projects/bird_song_classifier/1DCNN_inference_train_cm.png" alt="1D CNN inference train confusion matrix">
    </div>
    <div class="col-md-6 d-flex flex-column align-items-center justify-content-center">
      <img class="img-fluid" src="/assets/img/projects/bird_song_classifier/1DCNN_inference_test_cm.png" alt="1D CNN inference test confusion matrix">
    </div>
  </div>
  <p class='mt-3'>Compared to the original training results, we can see that the model performed slightly better on the test set (compared to the validation set) for barswa and comsan and slightly worse for eaywag1, although the difference is negligible. The test result is comparable to that of the validation result, suggesting the model is generalizing reasonabily well to the unseen test set. The overall test accuracy of 91% is also reasonabily high, as compared to the baseline.</p>

  <!-- C1. GRU RNN -->
  <h5 class='mt-3 mb-3'><strong>C1. GRU RNN</strong></h5>
  <p>Similar to what I did for inference on 1D CNN, I combined the training and validation set to one training dataset to train the model one last time before running inference using the GRU RNN model.Following the best performing GRU RNN training model, I used 20 MFCC + 1 RMS without continents as the main features with rating as sample weights, using the same hyperparameters as the hypertuned training model. The model was trained for 58 epochs with no callback. Once the model is done training, I ran inference on the test set using the same features. Below summarized is the resulting classification report and the confusion matrix.</p>
  <div class="row">
    <div class="col-md-6 d-flex flex-column align-items-center justify-content-center">
      <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/GRU_inference_train_report.png" alt="GRU RNN inference train classification report">
    </div>
    <div class="col-md-6 d-flex flex-column align-items-center justify-content-center">
      <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/GRU_inference_test_report.png" alt="GRU RNN inference test classification report">
    </div>
  </div>
  <div class="row">
    <div class="col-md-6 d-flex flex-column align-items-center justify-content-center">
      <img class="img-fluid" src="/assets/img/projects/bird_song_classifier/GRU_inference_train_cm.png" alt="GRU RNN inference train confusion matrix">
    </div>
    <div class="col-md-6 d-flex flex-column align-items-center justify-content-center">
      <img class="img-fluid" src="/assets/img/projects/bird_song_classifier/GRU_inference_test_cm.png" alt="GRU RNN inference test confusion matrix">
    </div>
  </div>
  <p class='mt-3'>The result is similar to that of the 1D CNN inference results, but with slightly higher overall accuracy and better performance across all three species. This is expected as RNNs are generally better suited for sequence processing tasks, whereas CNNs are generally better suited for image classification tasks. Audios, a form of digital signal, are sequential in nature. Although, by converting audio to spectrograms, therefore converting a digital signal processing task to a computer task seem to also yield promissing results. It is notable that CNNs are faster than RNNs, so perhaps the slight reduction in performance is affordable considering the potential in efficieny gain, although more thorough analysis should be performed to draw meaningful conclusions. The overall test accuracy of 93% is also reasonabily high, as compared to the baseline and as compared to all other models experimented.</p>
</div>

<!-- LIMITATIONS -->
<div class='mb-5' id='limitations'>
  <h3 class='mb-3'><u>LIMITATIONS</u></h3>
  <p>The biggest limitation of the project is that I only selected three of the most common bird species for the classification task, therefore, the models are not robust towards other bird species. I also did not perform ablation study for all models experimented, therefore, the model performance could be further improved if more comprehensive hypertuning were performed.</p>
  <p>Another notable limitation pertains to the fairness of the original dataset. The map view (generated by another teammember) below shows the geographical distribution of the samples, with each dot representing one sample and each color corresponding to a specific species. We can see that majority of the samples are sourced from Europe, closely followed by South Afric, while birds from other regions of the world (Americas, Oceania, Asia) are under-represented. We could improve fairness by adjusting the data gathering process or sample weighting scheme to create more generalizable bird song classification models for birds from all over the world.</p>
  <img class="img-fluid" src="/assets/img/projects/bird_song_classifier/distribution.png" alt="Bird sample distribution map">
</div>

<!-- GITHUB -->
<div class='mb-5' id='github'>
  <h3 class='mb-3'><u>GITHUB</u></h3>
  <p>Please see my <a href="https://github.com/rachlllg/Project_Bird-Song-Classifier-with-Machine-Learning">GitHub</a> for the code for the project.</p>
</div>