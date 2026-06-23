# Provenance Table Structure and Statistics

This document describes what each field in `mapping_provenance.csv` contains and provides a statistics snapshot.

## Provenance Table Columns


- `provenance_row_id`: sequential row identifier in the format `prov-0001`, `prov-0002`, etc.
- `form_name`: survey name.
- `field`: source field type (for example `observation_source_value` or `condition_source_value`).
- `source_value`: the source value. This can be a field-like value (especially for observations which will have the value in the field mapped to value_as_string) or a source data value (for conditions, for example).
- `source_concept_name`: source concept name text associated with the source value.
- `standard_concept_id`: OMOP standard concept id for the source value. This is what populates event-table `<domain>_concept_id` fields (for example, `observation_source_value` -> `observation_concept_id`).
- `source_concept_code`: source code for the source-side concept (Athena source code or selected source vocabulary code, such as MONDO). This is used in source-side concept id/code population workflows (for example, `observation_source_concept_id`).
- `source_lookup_uri`: source URI, populated only for MONDO-coded source concepts (`source_concept_code` starts with `MONDO:`), using `http://purl.obolibrary.org/obo/MONDO_<id>`.
- `target_omop_uri`: Athena OMOP URI, populated only when `standard_concept_id` is present, using `https://athena.ohdsi.org/search-terms/terms/<standard_concept_id>`.


## Statistics Snapshot (`mapping_provenance.csv`)

- Total rows: **2,982**
- Rows with `standard_concept_id` populated: **1,669**
- Rows without `standard_concept_id`: **1,313**
- Rows with `source_lookup_uri` populated: **204**
- Rows with `target_omop_uri` populated: **1,669**
- Current mapped percentage: 56%


## Additional notes
- Source values that included a Unit were mapped to a more broad term. The OMOP entry for those will however be expanded with "unit_source_value" and "unit_concept_id". 
- The same applies to source values that contained a qualifier. 