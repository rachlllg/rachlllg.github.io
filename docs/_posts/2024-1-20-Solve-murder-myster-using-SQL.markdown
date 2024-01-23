---
layout: post
title:  "Solve Murder Mystery Using SQL"
description: Ever wonder what SQL can be used for? To solve murder mysteries of course! 🔪 Follow along this project and try your own hands at solving the case, with a twist at the end! 
categories: SQL DataAnalysis
---

<div class='m-3'>
<p>I have done a number of SQL-based projects as part of my school work, but due to academic integrity limitations (ie, not allowed to post my solutions), I cannot share these projects on my website. So what's the next best alternative? Solve a <a href='https://mystery.knightlab.com/'>murder mystery</a> using SQL of course! 🔪 All queries are written in SQLite, let's dive right in!</p>
<!-- Step 0 -->
<h4 class='mb-3'><u>Step 0</u></h4>
<p>Before I start querying the database, I want to get a better understanding of each of the table, particularly the <span class="text-info">PRIMARY KEY</span>, <span class="text-info">FOREIGN KEY</span>, and <span class="text-info">REFERENCES</span>.</p>
<h5>QUERY:</h5>
<pre class='csv-table'>
  <code>
    SELECT sql
    FROM sqlite_master
  </code>
</pre>
<h5>RESULT:</h5>
<img class="img-fluid" src="/assets/img/blogs/murder/table.png" alt="structure of each table">
<p class='mt-3'>It would also be helpful to take a look at the schema diagram to visually see how the tables are related to one another.</p>
<h5>DIAGRAM:</h5>
<img class="img-fluid" src="/assets/img/blogs/murder/diagram.png" alt="schema diagram of the tables">
<p class='mt-3'>Before I query the selected tables based on the criteria, I usually run below query on the table first to gain an idea of what the data in the table looks like so I can tailor my query to match the data in the table.</p>
<h5>QUERY:</h5>
<pre class='csv-table'>
  <code>
    SELECT *
    FROM table_name
    LIMIT 5
  </code>
</pre>
<!-- Step 1 -->
<h4 class='mb-3'><u>Step 1</u></h4>
<p>Now I'm ready to get started! I know that 'the crime was a <span class="text-info">​murder​</span> that occurred sometime on <span class="text-info">​Jan.15, 2018</span>​ and that it took place in <span class="text-info">​SQL City</span>', so I first queried the <span class="text-info">crime_scene_report</span> table to find the relevant report.</p>
<h5>QUERY:</h5>
<pre class='csv-table'>
  <code>
    SELECT *
    FROM crime_scene_report
    WHERE date = 20180115 AND city = 'SQL City' AND type = 'murder'
  </code>
</pre>
<h5>RESULT:</h5>
<img class="img-fluid" src="/assets/img/blogs/murder/crime_report.png" alt="crime_scene_report with details that match the criteria">
<p class='mt-3'>Based on the description of the crime report, I know there are two witnesses. I have the first name of one of the witness and the street address of both witnesses.</p>
<!-- Step 2 -->
<h4 class='mb-3'><u>Step 2</u></h4>
<p>Let's start with the witness named <span class="text-info">Annabel​</span> first by querying the <span class="text-info">person​</span> table based on the street name and the first name.</p>
<h5>QUERY:</h5>
<pre class='csv-table'>
  <code>
    SELECT *
    FROM person
    WHERE address_street_name = 'Franklin Ave' 
      AND name LIKE 'Annabel%'
  </code>
</pre>
<h5>RESULT:</h5>
<img class="img-fluid" src="/assets/img/blogs/murder/person_annabel.png" alt="personal information of witness Annabel">
<p></p>
<!-- Step 3 -->
<h4 class='mb-3'><u>Step 3</u></h4>
<p>I can then use <span class="text-info">Annabel</span>'s id from the <span class="text-info">person</span> table to find the interview transcript from the <span class="text-info">interview</span> table. For this query, I used a subquery within the main query to directly get <span class="text-info">Annabel</span>'s id from the <span class="text-info">person</span> table.</p>
<h5>QUERY:</h5>
<pre class='csv-table'>
  <code>
    SELECT *
    FROM interview
    WHERE person_id = (SELECT id
                        FROM person
                        WHERE address_street_name = 'Franklin Ave' 
                        AND name LIKE 'Annabel%')
  </code>
</pre>
<h5>RESULT:</h5>
<img class="img-fluid" src="/assets/img/blogs/murder/transcript_annabel.png" alt="interview transcript of witness Annabel">
<p class='mt-3'>Looks like <span class="text-info">Annabel</span> identified the killer as someone from her gym and the killer went to the gym on January 9th.</p>
<!-- Step 4 -->
<h4 class='mb-3'><u>Step 4</u></h4>
<p>Since the <span class="text-info">get_fit_now_check_in</span> table does not include additional member information, I joined the <span class="text-info">get_fit_now_check_in</span> table with the <span class="text-info">get_fit_now_member</span> table to more easily identify the members that used the gym on January 9th.</p>
<h5>QUERY:</h5>
<pre class='csv-table'>
  <code>
    SELECT *
    FROM get_fit_now_check_in AS ci
    JOIN get_fit_now_member AS mbr
    ON ci.membership_id = mbr.id
    WHERE ci.check_in_date = 20180109
  </code>
</pre>
<h5>RESULT:</h5>
<img class="img-fluid" src="/assets/img/blogs/murder/joined_mbr_ci.png" alt="interview transcript of witness Annabel">
<p class='mt-3'>We can see that <span class="text-info">Annabel</span> indeed went to the gym on January 9th, as she claimed in the interview. The suspects would be the gym members that attended gym during the same period as <span class="text-info">Annabel</span>.</p>
<!-- Step 5 -->
<h4 class='mb-3'><u>Step 5</u></h4>
<p>To find the suspects that attended the gym during the same period as <span class="text-info">Annabel</span>, I added on to the query from previous step to filter by the <span class="text-info">check_in_time</span> and <span class="text-info">check_out_time</span>. For the time period to overlap, they must have left the gym after <span class="text-info">Annabel</span> entered and entered the gym before <span class="text-info">Annabel</span> left. Since the tables are relatively small in this example, we can visually inspect the result from the previous query and confirm the result is reasonable. To make the query more readable, I try to use common table expressions (CTEs) for most of the queries that involved more than one subquery.</p>
<h5>QUERY:</h5>
<pre class='csv-table'>
  <code>
    WITH gym_named AS (
      SELECT *
      FROM get_fit_now_check_in AS ci
      JOIN get_fit_now_member AS mbr
      ON ci.membership_id = mbr.id
      WHERE ci.check_in_date = 20180109
      ),
    annabel AS (
      SELECT *
      FROM gym_named
      WHERE check_in_date = 20180109 
        AND name LIKE 'Annabel%'
      )
    SELECT * 
    FROM gym_named
    WHERE check_out_time >= (SELECT check_in_time FROM annabel) 
      AND check_in_time <= (SELECT check_out_time FROM annabel)
  </code>
</pre>
<h5>RESULT:</h5>
<img class="img-fluid" src="/assets/img/blogs/murder/annabel_suspects.png" alt="potential suspects who attended gym at the same time as Annabel">
<p class='mt-3'>We can identify two suspects, <span class="text-info">Joe</span> and <span class="text-info">Jeremy</span>.</p>
<!-- Step 6 -->
<h4 class='mb-3'><u>Step 6</u></h4>
<p>Before moving on to finding out more information about the two suspects, let's identify the other witness and find out what the other witness has to say about the case first. I know the witness lives at the last house of <span class="text-info">Northwestern Dr</span> so assuming the house numbers are in increasing order, we can find out the witness from the <span class="text-info">person</span> table.</p>
<h5>QUERY:</h5>
<pre class='csv-table'>
  <code>
    SELECT * 
    FROM person
    WHERE address_number = (SELECT MAX(address_number)
                            FROM person
                            WHERE address_street_name = 'Northwestern Dr')
      AND address_street_name = 'Northwestern Dr'
  </code>
</pre>
<h5>RESULT:</h5>
<img class="img-fluid" src="/assets/img/blogs/murder/person_morty.png" alt="info on the witness that lives at the last house of Northwestern Dr">
<p class='mt-3'>We now know who the other witness is, we can find out what this witness said in their interview.</p>
<!-- Step 7 -->
<h4 class='mb-3'><u>Step 7</u></h4>
<p>Similar to what was done for <span class="text-info">Annabel</span>, I queried the <span class="text-info">interview</span> table to find the interview transcript for <span class="text-info">Morty</span>. Now we have two nested subqueries which makes the query a little hard to read, but since the logic here is pretty simple, I just left it as is.</p>
<h5>QUERY:</h5>
<pre class='csv-table'>
  <code>
    SELECT *
    FROM interview
    WHERE person_id = (SELECT id
                        FROM person
                        WHERE address_number = (SELECT MAX(address_number)
                                                FROM person
                                                WHERE address_street_name = 'Northwestern Dr')
                          AND address_street_name = 'Northwestern Dr')
  </code>
</pre>
<h5>RESULT:</h5>
<img class="img-fluid" src="/assets/img/blogs/murder/transcript_morty.png" alt="interview transcript of witness morty">
<p class='mt-3'>From the transcript, we can confirm that the witness goes to the gym and is a gold member. We also know his membership number and his license plate number.</p>
<!-- Step 8 -->
<h4 class='mb-3'><u>Step 8</u></h4>
<p>Let's now cross check the information <span class="text-info">Morty</span> provided with the information we found out based on what <span class="text-info">Annabel</span> provided. </p>
<h5>QUERY:</h5>
<pre class='csv-table'>
  <code>
    WITH gym_named AS (
      SELECT *
      FROM get_fit_now_check_in AS ci
      JOIN get_fit_now_member AS mbr
      ON ci.membership_id = mbr.id
      WHERE ci.check_in_date = 20180109
      ),
    annabel AS (
      SELECT *
      FROM gym_named
      WHERE check_in_date = 20180109 AND name LIKE 'Annabel%'
      )
    SELECT * 
    FROM gym_named
    WHERE check_out_time >= (SELECT check_in_time FROM annabel) 
      AND check_in_time <= (SELECT check_out_time FROM annabel)
      AND membership_status = 'gold'
      AND membership_id LIKE '48Z%'
  </code>
</pre>
<h5>RESULT:</h5>
<img class="img-fluid" src="/assets/img/blogs/murder/morty_suspects.png" alt="annabel suspects that also match morty gym description">
<p class='mt-3'>Unfortunately, knowing the gym information did not help us narrow down the suspects as both <span class="text-info">Joe</span> and <span class="text-info">Jeremy</span> are <span class="text-info">gold</span> members with <span class="text-info">membership_id</span> starting with <span class="text-info">48Z</span>. But at least we now know the <span class="text-info">Annabel</span> is not a suspect.</p>
<!-- Step 9 -->
<h4 class='mb-3'><u>Step 9</u></h4>
<p>Luckily, <span class="text-info">Morty</span> gave us one more clue, which is the suspect's car license plate. We can find the <span class="text-info">license_id</span> from the <span class="text-info">person</span> table and then find the <span class="text-info">plate_number</span> from the <span class="text-info">drivers_license</span> table. Since the <span class="text-info">drivers_license</span> table does not include the name of the person, I join the <span class="text-info">drivers_license</span> table to the <span class="text-info">person</span> table to create the <span class="text-info">dl_named</span> table to easily identify the name of the killer.</p>
<h5>QUERY:</h5>
<pre class='csv-table'>
  <code>
    WITH gym_named AS (
      SELECT *
      FROM get_fit_now_check_in AS ci
      JOIN get_fit_now_member AS mbr
      ON ci.membership_id = mbr.id
      WHERE ci.check_in_date = 20180109
      ),
    annabel AS (
      SELECT *
      FROM gym_named
      WHERE check_in_date = 20180109 
        AND name LIKE 'Annabel%'
      ),
    gym_suspects AS (
      SELECT *
      FROM gym_named
      WHERE check_out_time >= (SELECT check_in_time FROM annabel) 
        AND check_in_time <= (SELECT check_out_time FROM annabel)
        AND membership_status = 'gold'
        AND membership_id LIKE '48Z%'
      ),
    dl_named AS (
      SELECT ps.id AS person_id, ps.name, ps.license_id, ps.ssn, dl.* 
      FROM person AS ps
      JOIN drivers_license AS dl
      ON ps.license_id = dl.id
      )
    SELECT * 
    FROM dl_named
    WHERE person_id IN (SELECT person_id 
                        FROM gym_suspects)
      AND plate_number LIKE '%H42W%'
  </code>
</pre>
<h5>RESULT:</h5>
<img class="img-fluid" src="/assets/img/blogs/murder/killer.png" alt="killer">
<p class='mt-3'>Voila!</p>
<!-- Step 10 -->
<h4 class='mb-3'><u>Step 10</u></h4>
<p>After finding out the killer, I used the provided script to check my answers, and sure enough, I had correctly identified the killer. But wait there is more!</p>
<h5>RESULT:</h5>
<img class="img-fluid" src="/assets/img/blogs/murder/result.png" alt="result">
<p class='mt-3'>So I made a small change to the previous query to find the transcript from the killer.</p>
<h5>QUERY:</h5>
<pre class='csv-table'>
  <code>
    WITH gym_named AS (
      SELECT *
      FROM get_fit_now_check_in AS ci
      JOIN get_fit_now_member AS mbr
      ON ci.membership_id = mbr.id
      WHERE ci.check_in_date = 20180109
      ),
    annabel AS (
      SELECT *
      FROM gym_named
      WHERE check_in_date = 20180109 
        AND name LIKE 'Annabel%'
      ),
    gym_suspects AS (
      SELECT *
      FROM gym_named
      WHERE check_out_time >= (SELECT check_in_time FROM annabel) 
        AND check_in_time <= (SELECT check_out_time FROM annabel)
        AND membership_status = 'gold'
        AND membership_id LIKE '48Z%'
      ),
    dl_named AS (
      SELECT ps.id AS person_id, ps.name, ps.license_id, ps.ssn, dl.* 
      FROM person AS ps
      JOIN drivers_license AS dl
      ON ps.license_id = dl.id
      ),
    killer AS (
      SELECT * 
      FROM dl_named
      WHERE person_id IN (SELECT person_id 
                          FROM gym_suspects)
        AND plate_number LIKE '%H42W%'
      )
    SELECT *
    FROM interview
    WHERE person_id = (SELECT person_id 
                        FROM killer)
  </code>
</pre>
<h5>RESULT:</h5>
<img class="img-fluid" src="/assets/img/blogs/murder/killer_transcript.png" alt="interview transcript of the killer">
<p class='mt-3'>Evidently, this was a hired murder! 😲 The killer's transcript provided very detailed description of the woman that hired him, so let's try to find out who the 'real' killer is!</p>
<!-- Step 11 -->
<h4 class='mb-3'><u>Step 11</u></h4>
<p>We know that we are looking for someone who is 1) female, 2) has red hair, 3) drives Tesla Model S, 4) is around 65" - 67" tall, and attended the SQL Symphony Concert 3 times in December 2017.</p>
<p>I first identified the suspects from the <span class="text-info">drivers_license</span> table matching the physical and car description of the suspect, joining the <span class="text-info">drivers_license</span> table with the <span class="text-info">person</span> table so the <span class="text-info">person_id</span> can be easily queried and the <span class="text-info">name</span> can be easily identified.</p>
<p>I then identified the relevant concerts from the <span class="text-info">facebook_event_checkin</span> table, grouped the table by the <span class="text-info">person_id</span> to find the number of attendance for each person to identify suspects that have attended the event 3 times.</p>
<p>Finally, the 'real' killer is the only suspect in both the <span class="text-info">dl_suspects</span> and <span class="text-info">concert_suspects</span> subquery tables.</p>
<h5>QUERY:</h5>
<pre class='csv-table'>
  <code>
    WITH dl_suspects AS (
      SELECT ps.id AS person_id, ps.name, ps.license_id, ps.ssn, dl.* 
      FROM person AS ps
      JOIN drivers_license AS dl
      ON ps.license_id = dl.id
      WHERE dl.gender = 'female'
        AND dl.hair_color = 'red'
        AND dl.car_make = 'Tesla'
        AND dl.car_model = 'Model S'
        AND 65 <= dl.height <= 67
      ),
    concert AS (
      SELECT *
      FROM facebook_event_checkin
      WHERE 20171201 <= date <= 20171231
      AND event_name LIKE '%SQL Symphony Concert%'
      ),
    concert_suspects AS (
      SELECT person_id, COUNT(*) AS num_attendance
      FROM concert
      GROUP BY person_id
      HAVING num_attendance = 3
      )
    SELECT *
    FROM dl_suspects
    WHERE person_id IN (SELECT person_id 
                        FROM concert_suspects)
  </code>
</pre>
<h5>RESULT:</h5>
<img class="img-fluid" src="/assets/img/blogs/murder/real_killer.png" alt="the 'real' killer that paid for the murder">
<p></p>
<img class="img-fluid" src="/assets/img/blogs/murder/final.png" alt="final result">
<p class='m-3'>At last! 🍾🍾🍾</p>
<p>That was really fun! Of course, the dataset used in this project was very small and quite easy to work through, real life data would be much larger and much messier!</p>
<p>Also see my <a href='https://github.com/rachlllg/Blog_Solve-Murder-Mystery-using-SQL'>GitHub</a> for the full list of SQL commands used.</p>
</div>
