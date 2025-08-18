"""
Database setup script for TruthLens Evidence schema.

This script provides utilities to set up the database tables for storing evidence data.
It supports both PostgreSQL (with pgvector) and BigQuery.
"""

import os
import sys
from typing import Optional, Dict, Any
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.append(str(Path(__file__).parent.parent))

from schemas.database import (
    create_all_postgres_schemas,
    create_all_bigquery_schemas,
    get_postgres_schema,
    get_bigquery_schema
)


def setup_postgresql(connection_string: Optional[str] = None, output_file: Optional[str] = None):
    """
    Set up PostgreSQL database with TruthLens schemas.
    
    Args:
        connection_string: PostgreSQL connection string (if None, only generates SQL)
        output_file: Optional file to save SQL statements
    """
    print("Setting up PostgreSQL database schemas...")
    
    # Get all PostgreSQL schemas
    schemas = create_all_postgres_schemas()
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write("-- TruthLens PostgreSQL Schema Setup\n")
            f.write("-- Generated automatically\n\n")
            
            for i, schema in enumerate(schemas, 1):
                f.write(f"-- Table {i}\n")
                f.write(schema)
                f.write("\n\n")
        
        print(f"SQL schemas saved to: {output_file}")
    
    if connection_string:
        try:
            import psycopg2
            from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
            
            # Connect to PostgreSQL
            conn = psycopg2.connect(connection_string)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            print("Connected to PostgreSQL. Creating tables...")
            
            # Execute each schema
            for i, schema in enumerate(schemas, 1):
                print(f"Creating table {i}...")
                cursor.execute(schema)
            
            print("All PostgreSQL tables created successfully!")
            
            cursor.close()
            conn.close()
            
        except ImportError:
            print("psycopg2 not installed. Install with: pip install psycopg2-binary")
            print("Only SQL generation is available.")
        except Exception as e:
            print(f"Error setting up PostgreSQL: {e}")
    else:
        print("No connection string provided. SQL schemas generated for manual execution.")


def setup_bigquery(project_id: str, dataset_id: str, output_file: Optional[str] = None):
    """
    Set up BigQuery dataset with TruthLens schemas.
    
    Args:
        project_id: Google Cloud project ID
        dataset_id: BigQuery dataset ID
        output_file: Optional file to save schema definitions
    """
    print(f"Setting up BigQuery dataset: {project_id}.{dataset_id}")
    
    # Get all BigQuery schemas
    schemas = create_all_bigquery_schemas()
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write("-- TruthLens BigQuery Schema Setup\n")
            f.write("-- Generated automatically\n\n")
            
            for table_name, schema in schemas.items():
                f.write(f"-- Table: {table_name}\n")
                f.write(f"CREATE TABLE `{project_id}.{dataset_id}.{table_name}` (\n")
                
                fields = []
                for field in schema['fields']:
                    field_def = f"  `{field['name']}` {field['type']}"
                    if field['mode'] == 'REQUIRED':
                        field_def += " NOT NULL"
                    elif field['mode'] == 'REPEATED':
                        field_def += " ARRAY"
                    fields.append(field_def)
                
                f.write(",\n".join(fields))
                f.write("\n);\n\n")
        
        print(f"BigQuery schemas saved to: {output_file}")
    
    print("BigQuery schema definitions generated.")
    print("Use the BigQuery console or bq command-line tool to create tables.")


def generate_schema_files():
    """Generate schema files for both PostgreSQL and BigQuery."""
    print("Generating schema files...")
    
    # Create output directory
    output_dir = Path("database_schemas")
    output_dir.mkdir(exist_ok=True)
    
    # Generate PostgreSQL schemas
    postgres_file = output_dir / "postgresql_setup.sql"
    setup_postgresql(output_file=str(postgres_file))
    
    # Generate BigQuery schemas
    bigquery_file = output_dir / "bigquery_setup.sql"
    setup_bigquery("your-project-id", "truthlens_dataset", output_file=str(bigquery_file))
    
    print(f"\nSchema files generated in: {output_dir}")
    print(f"- PostgreSQL: {postgres_file}")
    print(f"- BigQuery: {bigquery_file}")


def main():
    """Main function to run database setup."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Set up TruthLens database schemas")
    parser.add_argument(
        "--postgres", 
        help="PostgreSQL connection string (e.g., postgresql://user:pass@host:port/db)"
    )
    parser.add_argument(
        "--postgres-output", 
        help="Output file for PostgreSQL SQL statements"
    )
    parser.add_argument(
        "--bigquery-project", 
        default="your-project-id",
        help="BigQuery project ID"
    )
    parser.add_argument(
        "--bigquery-dataset", 
        default="truthlens_dataset",
        help="BigQuery dataset ID"
    )
    parser.add_argument(
        "--bigquery-output", 
        help="Output file for BigQuery schema definitions"
    )
    parser.add_argument(
        "--generate-files", 
        action="store_true",
        help="Generate schema files for both databases"
    )
    
    args = parser.parse_args()
    
    if args.generate_files:
        generate_schema_files()
        return
    
    if args.postgres:
        setup_postgresql(args.postgres, args.postgres_output)
    
    if args.bigquery_project and args.bigquery_dataset:
        setup_bigquery(args.bigquery_project, args.bigquery_dataset, args.bigquery_output)
    
    if not any([args.postgres, args.bigquery_project, args.generate_files]):
        print("No database specified. Use --help for options.")
        print("\nExample usage:")
        print("  # Generate schema files only")
        print("  python setup_database.py --generate-files")
        print("\n  # Set up PostgreSQL")
        print("  python setup_database.py --postgres 'postgresql://user:pass@localhost:5432/truthlens'")
        print("\n  # Set up BigQuery")
        print("  python setup_database.py --bigquery-project my-project --bigquery-dataset truthlens")


if __name__ == "__main__":
    main()
