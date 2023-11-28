---
layout: post
title:  "How well do LLMs understand grammar?"
description: A question many of us have wondered, do LLMs actually understand grammar? Follow along this blog post for a light-hearted discussion on this very topic.
categories: Python MachineLearning NLP
---

<div class='m-3'>
<p>In my project <a href='/project/2023-Fine_Tune_LLM_for_Grammatical_Classification/'>'Fine Tune LLM for Grammatical Classification'</a>, I fine-tuned an LLM (RoBERTa) on the CoLA dataset using Tensorflow to create a grammatical acceptability classifier. The pre-trained Roberta model and the tokenizer are sourced from HuggingFace 🤗, but the fine-tuning was done manually in Google Colab using a T4 GPU. By tweaking the hyperparameters, I matched the performance to the original published Roberta paper on the CoLA validation set, and by further tweaking the grammatical threshold, I improved the performance by 3 points. The best-performing model achieved a validation accuracy of 88% with an MCC of 71.10.</p>
<p>In this blog post, I will use this model to evaluate real-life sentences to see how well the model can distinguish grammatical vs ungrammatical sentences. Disclaimer: This blog post is not intended to be a comprehensive discussion on LLM and their ability to interpret grammar. For comprehensive research on this topic, refer to published literature!</p>
<p>To follow along, I invite you to download and save the <a href='https://drive.google.com/file/d/1-SkcjKRqJcINL48MLQxBV2ik7MvvNCxC/view?usp=drive_link'>model weights</a> to your Google Drive, then open <a href='https://colab.research.google.com/drive/12ZQim9oQwRbcJwxwBEz9P1WJ3SF79Agl?usp=sharing'>this Google Colab notebook</a> and update the path to reload the trained weights. The notebook is configured to use a T4 GPU but not strictly necessary for inference. It will be slower without a GPU but still doable. If you are curious about the model specifications, I invite you to read through my <a href='/project/2023-Fine_Tune_LLM_for_Grammatical_Classification/'>project</a>, I will omit detailed discussion on the model specifications in this blog post.</p>
<p>After loading the model, we can pass in a few example sentences to extract the grammatical scores. Although the model is a binary classifier (grammatical vs ungrammatical), it outputs the probability of the input sentence before passing it to the final sigmoid layer, which converts it to a binary output. To extract the score, we bypass the final sigmoid layer and directly extract the probability of the input sentences.</p>
<p>In our example, the ungrammatical sentences are highlighted in green.</p>
<pre class="csv-table">
    <table>
      <thead>
        <tr>
          <th>Index</th>
          <th>Sentence</th>
          <th>Score</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>1</td>
          <td>I like to eat apples.</td>
          <td>0.9997229</td>
        </tr>
        <tr>
          <td>2</td>
          <td>I like eating apples.</td>
          <td>0.9996381</td>
        </tr>
        <tr>
          <td>3</td>
          <td>I love to eat apples.</td>
          <td>0.99985325</td>
        </tr>
        <tr style="background-color: #99FF99;">
          <td>4</td>
          <td>I like eat apples.</td>
          <td>0.03367024</td>
        </tr>
        <tr style="background-color: #99FF99; color:red;">
          <td>5</td>
          <td>I lake to eat apples.</td>
          <td>0.9966072</td>
        </tr>
        <tr style="background-color: #99FF99;">
          <td>6</td>
          <td>I love to ate apples.</td>
          <td>0.03415065</td>
        </tr>
        <tr style="background-color: #99FF99;">
          <td>7</td>
          <td>I love to eat ables.</td>
          <td>0.24197567</td>
        </tr>
      </tbody>
    </table>
  </pre>
<p>The model was highly confident in the three grammatical sentences, giving all sentences probability above 0.999. The model recognized three of the four ungrammatical sentences, giving them low probabilities.</p>
<p>Curiously though, the model was highly confident in the fifth sentence, having given it a high probability of above 0.996, even though the sentence is clearly ungrammatical, with 'lake' a NOUN in the place of a VERB. 🤔</p>
<p>You probably noted that I referred to the probability as 'score' earlier, which is probably not the most accurate word to use in this context. Although the model was trained to 'recognize' grammar, as an LLM, the probability is in fact the likelihood of the sentence. In other words, when the model sees 'I love to eat ables.', the output of 0.24 indicates the likelihood of the sentence, where 'ables' is the NOUN that follows 'eat'. Of course, this is unlikely.</p>
<p>Curiously, the model scored 'I lake to eat apples.' high, indicating the model believes this sentence is highly likely, even though it should not be likely at all. However, we do need to remember that our training data was highly skewed, where 70% of the training data was grammatical. There is a possibility the model learned to recognize grammatical sentences better than its ungrammatical counterparts due to the skewness in the training data. We also have to remember our model only scored ~88% in accuracy, suggesting the model is correct only 88% of the time. Of course, since the training data was not balanced, accuracy is a poorly chosen metric to begin with. Hence MCC is the metric used to evaluate the CoLA dataset in the <a href='https://gluebenchmark.com/'>GLUE benchmark</a>.</p>
<p>The incorrectly classified example above is an example of False Positive, where the true label is negative while the model predicted it to be positive. The counterpart of False Positive is False Negative, where the true label is positive while the model predicted it to be negative.</p>
<p>So, onto the question I posed at the beginning: "Do LLMs actually understand grammar?" Well, here are my two cents: 'Understand' is a strong word that's difficult to define. Do our dogs 'understand' the word 'sit'? Or are they just doing it out of reflex because they've associated the action of sitting with treats? We know that LLMs are mimicking what they have been exposed to, which to some extent I'd argue is a shallow level of 'understanding'. That is how we learn languages as a baby after all, we hear what our mothers say and try to utter the same sound. But do LLMs really understand grammar in the sense that a NOUN follows a VERB and an ADJ is used to describe a NOUN? Perhaps not really. They just know certain words are commonly associated with others in the larger corpus.</p>
<p>Of course, this is a controversial and complex question, and I would love to hear your thoughts on this as well! And don't forget to try out the GAC model with your own examples!</p>