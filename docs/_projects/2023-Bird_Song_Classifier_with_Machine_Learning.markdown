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
  <div class="row mb-4">
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
  <div class="row mb-4">
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
  <div class="row mb-4">
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
  <ul class='mb-4'>
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
    <div class="col-md-6 mb-4 d-flex align-items-center justify-content-center">
      <iframe src="https://www.youtube.com/embed/2U3mY00rDeA?si=G7tSgU-Ld0Cnmovz" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    </div>
  </div>
  <ol class='mb-4'>
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
  <ul class='mb-4'>
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
  <p>To overcome the class imbalance in total duration, during train/validation split, I downsampled the oversampled species in training set and put the rest in validation set. By doing this, the training set is now balanced, which would allow the models to learn features from the three classes equally well during training.</p>
  <img class="img-fluid mb-3" src="/assets/img/projects/bird_song_classifier/train_duration.png" alt="total duration by species in training set">
  <p>The total duration in the validation set is imbalanced after splitting to train/val sets, as it should be representative of the potential imbalance in the test set.</p>
  <img class="img-fluid mb-5" src="/assets/img/projects/bird_song_classifier/val_duration.png" alt="total duration by species in validation set">
  <!-- Create Class Methods -->
  <h5 class='mb-3'><strong>B. Create Class Methods</strong></h5>
      <p>Even though only 3 species were selected for the project, the audio features for the training and validation set are still too large to fit into memory of Google Colab (the free version only provides 12.7GB RAM). I therefore created two class methods which allowed me to manage the memory usage more efficiently.</p>
  <div class="row">
    <div class="col-md-6">
      <p class='mb-4'>The Framed class is used to frame the audios (split audios of varying length to set lengths clips with or without augmentation, and with or without overlapping). The Extraction class is used to extract the audio features and labels from each of the framed clips (with or without normalization and/or average pooling) in a shape that's ready to be passed into the models. The code used to create and test the class methods can be found in the b.class_methods.ipynb file in the 3.model_prep folder of the GitHub repo.</p>
    </div>
    <div class="col-md-6 mb-3 d-flex flex-column align-items-center justify-content-center">
      <iframe src="https://www.youtube.com/embed/44BFY5bFt10?si=jVPH70Kzb0EfFWgp" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
      <p class='text-center'>Part 1 (Train/Val split & class methods)</p>
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
  <p class='mb-4'></p>
  <!-- Extract Features & Labels -->
  <h5 class='mb-3'><strong>D. Extract Features & Labels</strong></h5>
  <p>Once the framed audios have been extracted, I then used the Extraction class (discussed above) to extract the various features with below specifications (all numeric features were normalized), and then saved the extracted features to disk (using pickle) for future use. The numbers in brackets indicate the number of each feature extracted from each audio. The code used to extract and save the features can be found in the d.extract_features_labels folder in the 3.model_prep folder of the GitHub repo.</p>
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
    <div class="col-md-6 mb-3 d-flex flex-column align-items-center justify-content-center">
      <iframe src="https://www.youtube.com/embed/4rdYVgKYAE8?si=H3-T70HWETzls1gg" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
      <p class='text-center'>Part 2 (Extract framed audio & features)</p>
    </div>
  </div>
  <p class='mb-4'></p>
</div>

<!-- TRAINING -->
<div class='mb-5' id='training'>
  <h3 class='mb-3'><u>TRAINING</u></h3>
  <!-- A1. Baseline -->
  <h5 class='mb-3'><strong>A1. Baseline</strong></h5>
  <p>Each species represent 1/3 of the total duration in the training set, if one were to randomly guess the species, we would expect a 33% accuracy. This will serve as our baseline.</p>
  <p class='mb-4'>All notebooks for the models can be found in the 4.training folder in the GitHub repo.</p>
  <!-- B1. Ensemble - Random Forest -->
  <h5 class='mb-3'><strong>B1. Ensemble - Random Forest</strong></h5>
  <div class="row">
    <div class="col-md-6">
      <p>Ensemble is a machine learning technique that combined multiple models to one model. Random Forest, rooted from decision tree, is one of the ensemble techniques, where multiple decision trees are used to find the most optimal predictive feature at each 'branch'. I used the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html">RandomForestClassifier</a> from sklearn to implement the random forest models.</p>
    </div>
    <div class="col-md-6 mb-3 d-flex align-items-center justify-content-center">
      <iframe src="https://www.youtube.com/embed/dhUUCJntQrg?si=gME3Xs15kPjPb3MU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    </div>
  </div>
  <p>As mentioned earlier, many different audio features (such as MFCC, RMS, etc) could be extracted and used as features in machine learning models, so I implemented random forest models with 16 different combinations of various features to find the features with the strongest predictive power. I also experimented with different framed audio duration (3 seconds with 1 second overlap, 5 seconds with 2.5 seconds overlap, and 8 seconds with 4 seconds overlap) in case the audio frame duration made a difference in the models.</p>
  <p>Summarized below are the feature combinations from the models with the highest validation accuracy for each framed audio duration, with no augmentation. All models were generated with 50 estimators, with entropy as the criterion, and with combinations of normalized and average pooled 20 MFCC, 20 melspectrogram, 12 chroma, 1 RMS, 1 spectral Centroid, and/or 5 one-hot encoded continents.</p>
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
  <p>From this summary, we can see MFCC is predominently a better predictor than melspectrogram, and MFCC + continents are the strongest predictor among all combinations. The predictive power of RMS, chroma, and spectral centroid is not conclusive from this exercise alone, and even though framed audios with 8 seconds in length had the highest validation accuracy, the different to that of 5 seconds is not glaring so I would not conclude framed audios with 8 seconds in length are better than those with 5 seconds in length (at least not just yet). Notabily, framing audios to 3 seconds created more training sample counts, but the accuracy was actually lower than framing audios to longer durations. Going forward, I will only work with framing duration of 5 seconds or 8 seconds.</p>
  <p>I also tried applying random augmentation to the training samples for the 5 seconds and 8 seconds framed samples, but the highest valication accuracy was actually lower than those without augmentation. This is not to say augmentation will not help with model performance, but rather the augmentation technique may need to be revisited. I also tried increasing the number of melspectrogram to 128 (instead of 20), which did not make a difference in model performance.</p>
  <p class='mb-4'>For hypertuning the models, I changed the number of estimators to 40 and 80. Neither change made notable difference in the model performance. I did not hypertune the criterion as my research suggested entropy is generally the better criterion to use.</p>
  <!-- B2. Ensemble - XGBoost -->
  <h5 class='mb-3'><strong>B2. Ensemble - XGBoost</strong></h5>
  <p><a href="https://xgboost.readthedocs.io/en/stable/index.html#">XGBoost</a> is another ensemble machine learning technique that utilizes the gradient boosting framework to provide parallel tree boosting in decision tree algorithms. In the random forest notebooks, I used the Framed and Extraction classes for framing the audios and extracting the features, but starting from XGBoost, I will be directly using the features saved in .pkl format to improve efficiency.</p>
  <p>Similar to the random forest models, I experimented with different combinations of features from 5 seconds framed audios and 8 seconds framed audios, with and without augmentation. Summarized below are the feature combinations from the models with the highest validation accuracy for each framed audio duration. All models were generated with 100 estimators, with dart as the booster, and with combinations of normalized and average pooled 20 MFCC, 20 melspectrogram, 12 chroma, 1 RMS, 1 spectral Centroid, and/or 5 one-hot encoded continents.</p>
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
  <p>Similar to the findings from the random forest models, MFCC is predominently a better predictor than melspectrogram, and MFCC + chroma or MFCC + RMS together with continents are consistently the better predictor than any other feature combinations. The framed audios with 8 seconds in length again had higher validation accuracy than those with 5 seconds in length, and the random augmentation applied to the audios did not play a role in improving the models.</p>
  <p class='mb-4'>For hypertuning the models, I used <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV">GridSearchCV</a> from sklearn. The GridSearch was ran on the model with the same specifications as the highlighted model above. I also ran another model with the same specifications but replaced RMS with chroma. The GridSearch identified the best max depth at 6 with 200 estimators. However, the validation accuracy with this optimal hyperparameter setting did not improve over the original model, this is expected as the model is overfitted to the training data and already learned 100% of the features in the training data in the original model (with fewer estimators). To improve the model, more training data is likely needed.</p>
  <!-- C1. Support Vector Machine -->
  <h5 class='mb-3'><strong>C1. Support Vector Machine (SVM)</strong></h5>
  <p>Another tranditional machine learning algorithm commonly used for classification tasks is support vector machine (SVM), which is an algorithm used to identify a hyperplane that segregates/classifies the data points in an N-dimensional space. I used <a href='https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html'>SVC</a> from sklearn to implement the SVM models.</p>
  <p>I again experimented with different combinations of features from 5 seconds framed audios and 8 seconds framed audios, with and without augmentation. Summarized below are the feature combinations from the models with the highest validation accuracy for each framed audio duration. All models were generated with C=4 (regularization parameter), with rbf as the kernel, and with combinations of normalized and average pooled 20 MFCC, 20 melspectrogram, 12 chroma, 1 RMS, 1 spectral Centroid, and/or 5 one-hot encoded continents.</p>
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
  <p>The results are still consistent with the findings from the previous two models so I will omit detailed discussion here. Notabily SVM was much faster to run than XGBoost and the training results are less overfitted with comparable validation accuracy.</p>
  <p class='mb-4'>For hypertuning the models, I again used GridSearchCV from sklearn. The hypertuning did not improve the performance of the models.</p>
  <!-- D1. Logistic Regression -->
  <h5 class='mb-3'><strong>D1. Logistic Regression</strong></h5>
  <p></p>

  <h5 class='mb-4'><strong>NOTE: I ALREADY RAN BELOW LISTED MODELS ON A DIFFERENT (SIMILAR) DATASET, BUT THE LANGUAGE FOR THE WEBSITE IS NOT FINALIZED, SO PLEASE STAY TUNED AS I CONTINUE TO FINALIZED THIS EVERYDAY!</strong></h5>
  <!-- E1. Feed Forward Neural Network (FFNN) -->
  <h5 class='mb-3'><strong>E1. Feed Forward Neural Network (FFNN)</strong></h5>
  <p></p>

  <!-- F1. 1D Convolutional Neural Networks (1D-CNN) -->
  <h5 class='mb-3'><strong>F1. 1D Convolutional Neural Networks (1D-CNN)</strong></h5>
  <p></p>

  <!-- F2. 2D Convolutional Neural Networks (2D-CNN) -->
  <h5 class='mb-3'><strong>F2. 2D Convolutional Neural Networks (2D-CNN)</strong></h5>
  <p></p>

  <!-- G1. Recurrent Neural Networks - Long Short-Term Memory (LSTM RNN) -->
  <h5 class='mb-3'><strong>G1. Recurrent Neural Networks - Long short-term memory (LSTM RNN)</strong></h5>
  <p></p>

  <!-- H2. Recurrent Neural Networks - Gated Recurrent Unit (GRU RNN) -->
  <h5 class='mb-3'><strong>H2. Recurrent Neural Networks - Gated Recurrent Unit (GRU RNN)</strong></h5>
  <p></p>

  <!-- I1. Transformer -->
  <h5 class='mb-3'><strong>I1. Transformer</strong></h5>
  <p></p>

</div>


<!-- INFERENCE -->
<div class='mb-5' id='training'>
  <h3 class='mb-3'><u>INFERENCE</u></h3>
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


