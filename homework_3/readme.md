# DE300 Homework 3 — Airflow Movie Recommendation Pipeline

This project implements an Apache Airflow workflow that generates movie recommendations using interaction data stored in Amazon S3. The pipeline combines previously observed ratings with new partitions of data and produces recommendations for two types of users:

- **Cold user** – a user with no prior interactions  
- **Top user** – a randomly selected user from the top 5% most active users  

The workflow runs periodically and saves recommendation outputs to S3 using unique filenames to avoid overwriting previous results.

---

# System Architecture

The pipeline integrates several components.

## Amazon S3
- Stores rating data partitions
- Stores movie metadata
- Stores item embeddings
- Stores recommendation outputs

## Apache Airflow
- Schedules and orchestrates pipeline tasks
- Manages workflow dependencies
- Transfers intermediate results via XCom

## Python Recommendation Engine
- Uses item embeddings
- Computes cosine similarity for personalized recommendations
- Uses popularity-based ranking for cold users

---

# Pipeline Workflow

The Airflow DAG executes the following tasks.

## 1. Check Run Count

The pipeline reads the file `run_count.json`.

This file tracks how many times the pipeline has run.

If the run count is greater than 4, the pipeline stops execution. Otherwise, it continues with the workflow.

---

## 2. Download Data Partitions

The `download` task:

1. Downloads `run_count.json` from S3  
2. Determines how many partitions should be processed  
3. Downloads the relevant partitions (`partition_1.csv` to `partition_4.csv`)  
4. Merges them into a single dataset  
5. Uploads the merged dataset back to S3  

Each pipeline run processes progressively more interaction data.

---

## 3. Generate Recommendations

The `run_recommend` task performs recommendation generation.

Steps:

1. Download item embeddings (`full_embedding.npy`) from S3  
2. Normalize the embeddings  
3. Load rating data from the merged partitions  
4. Load movie metadata (`movies.dat`)  
5. Generate recommendations for a **Top user** and a **Cold user**

---

## Top User Recommendation

For the selected top user:

1. Identify a user whose number of interactions lies in the top 5%  
2. Extract their recent interactions  
3. Select movies with high ratings  
4. Compute a user embedding by averaging embeddings of watched movies  
5. Compute cosine similarity between the user vector and all item embeddings  
6. Recommend the top 5 unseen movies

---

## Cold User Recommendation

Cold users have no interaction history.

Recommendations are generated using popularity-based ranking by selecting the most frequently rated movies.

---

## Output Format

Each recommendation file contains the following fields:

- **User_Type** – Cold user or Top user  
- **User_ID** – ID of the selected user  
- **Last_Interaction_Time** – Timestamp of the user's most recent rating  
- **Num_Interactions** – Number of interactions observed  
- **Top5_Interaction_Cutoff** – Threshold defining top users  
- **Recommended_Movies** – List of recommended movies

---

## Upload Recommendations

The `upload` task:

1. Converts recommendations into CSV format  
2. Uploads the file to S3  

Output files follow the format:

`recs_<run_count>.csv`

Examples:

- recs_1.csv  
- recs_2.csv  
- recs_3.csv  
- recs_4.csv  

This naming convention ensures previous outputs are not overwritten.

---

## Update Run Count

The final task updates `run_count.json`.

The value is incremented by 1 after each DAG run so the next execution processes additional partitions.

---

# DAG Workflow

Pipeline order:

check_run_count → continue_dag → download → run_recommend → upload → update_count

If the run count exceeds the limit, the workflow stops.

---

# Scheduling

The DAG runs every **10 hours** using:

`pendulum.duration(hours=10)`

This allows the pipeline to periodically update recommendations as new interaction data becomes available.

---

# Technologies Used

- Python  
- Apache Airflow  
- Pandas  
- NumPy  
- Scikit-learn  
- AWS S3  
- Boto3  

---

# Running the Pipeline

1. Upload the DAG file to your Airflow or MWAA environment  
2. Upload required datasets to the S3 bucket  
3. Initialize `run_count.json` with value `1`  
4. Trigger the DAG in Airflow  

Each run will process additional partitions and generate new recommendation outputs.

---

# Example Output

Example record from a recommendation file:

User_Type: Top user  
User_ID: 123  
Last_Interaction_Time: 2024-03-05 12:31:02  
Num_Interactions: 45  
Recommended_Movies:  
1: Toy Story [Animation|Children|Comedy]  
50: Star Wars [Action|Adventure|Sci-Fi]
