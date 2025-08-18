-- TruthLens BigQuery Schema Setup
-- Generated automatically

-- Table: evidence_raw
CREATE TABLE `your-project-id.truthlens_dataset.evidence_raw` (
  `id` STRING NOT NULL,
  `claim_id` STRING NOT NULL,
  `source_type` STRING NOT NULL,
  `url` STRING NOT NULL,
  `domain` STRING NOT NULL,
  `title` STRING NOT NULL,
  `published_at` TIMESTAMP,
  `retrieved_at` TIMESTAMP NOT NULL,
  `language` STRING NOT NULL,
  `snippet` STRING,
  `full_text` STRING,
  `full_text_hash` STRING NOT NULL,
  `chunk_ids` STRING ARRAY,
  `support_label` STRING NOT NULL,
  `relevance_score` FLOAT64,
  `freshness_score` FLOAT64,
  `source_score` FLOAT64,
  `final_score` FLOAT64,
  `metadata` JSON,
  `created_at` TIMESTAMP NOT NULL,
  `updated_at` TIMESTAMP NOT NULL
);

-- Table: evidence_chunks
CREATE TABLE `your-project-id.truthlens_dataset.evidence_chunks` (
  `chunk_id` STRING NOT NULL,
  `evidence_id` STRING NOT NULL,
  `chunk_index` INT64 NOT NULL,
  `text` STRING NOT NULL,
  `text_hash` STRING NOT NULL,
  `embedding` FLOAT64 ARRAY,
  `language` STRING NOT NULL,
  `metadata` JSON,
  `created_at` TIMESTAMP NOT NULL
);

-- Table: claims_evidence
CREATE TABLE `your-project-id.truthlens_dataset.claims_evidence` (
  `claim_id` STRING NOT NULL,
  `evidence_id` STRING NOT NULL,
  `support_label` STRING NOT NULL,
  `relevance_score` FLOAT64,
  `freshness_score` FLOAT64,
  `source_score` FLOAT64,
  `final_score` FLOAT64,
  `confidence` FLOAT64,
  `annotator_id` STRING,
  `annotated_at` TIMESTAMP,
  `created_at` TIMESTAMP NOT NULL,
  `updated_at` TIMESTAMP NOT NULL
);

