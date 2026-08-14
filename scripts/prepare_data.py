import os
import argparse
import logging
import pandas as pd
import yaml

from src.utils.logging import setup_logging
from src.utils.seed import set_seed
from src.data.preprocessing import preprocess_dataframe
from src.data.splitting import prevent_leakage_and_split

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Preprocess and split DNA regulatory dataset.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file.")
    args = parser.parse_args()

    # Setup logging
    setup_logging(log_file="logs/prepare_data.log")
    logger.info("Initializing Data Preparation Pipeline...")

    # Load configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found at: {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    set_seed(seed)

    val_split = config["data"].get("val_split", 0.1)
    test_split = config["data"].get("test_split", 0.1)

    # --- PROCESS PROMOTER DATA ---
    logger.info("=========================================")
    logger.info("PROCESSING TASK: PROMOTER PREDICTION")
    logger.info("=========================================")
    
    prom_raw_dir = config["data"]["promoter"]["raw_dir"]
    prom_processed_dir = config["data"]["promoter"]["processed_dir"]
    
    # Check if raw files exist
    prom_files = ["train.csv", "dev.csv", "test.csv"]
    prom_exists = all(os.path.exists(os.path.join(prom_raw_dir, f)) for f in prom_files)
    
    if prom_exists:
        logger.info(f"Loading raw promoter files from: {prom_raw_dir}")
        dfs = []
        for filename in prom_files:
            file_path = os.path.join(prom_raw_dir, filename)
            df_part = pd.read_csv(file_path)
            logger.info(f"  Loaded {filename}: {len(df_part)} samples")
            dfs.append(df_part)
        
        # Merge to perform global deduplication and clean split
        merged_prom_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total merged promoter samples: {len(merged_prom_df)}")
        
        # Preprocess
        cleaned_prom_df, prom_stats = preprocess_dataframe(merged_prom_df)
        
        # Split deterministically (no leakage)
        train_df, val_df, test_df = prevent_leakage_and_split(
            cleaned_prom_df, 
            val_split=val_split, 
            test_split=test_split, 
            seed=seed
        )
        
        # Save splits
        os.makedirs(prom_processed_dir, exist_ok=True)
        train_df.to_csv(os.path.join(prom_processed_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(prom_processed_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(prom_processed_dir, "test.csv"), index=False)
        
        # Save stats JSON
        with open(os.path.join(prom_processed_dir, "preprocessing_stats.yaml"), "w") as f:
            yaml.dump(prom_stats, f)
            
        logger.info(f"Processed promoter data saved to: {prom_processed_dir}")
    else:
        logger.warning(f"Raw promoter files not found in {prom_raw_dir}. Skipping promoter processing.")

    # --- PROCESS TFBS DATA ---
    logger.info("=========================================")
    logger.info("PROCESSING TASK: TFBS PREDICTION")
    logger.info("=========================================")
    
    tf_raw_dir = config["data"]["tf"]["raw_dir"]
    tf_processed_dir = config["data"]["tf"]["processed_dir"]
    
    # TFBS dataset contains subfolders 0, 1, 2, 3, 4
    if os.path.exists(tf_raw_dir):
        subdirs = [d for d in os.listdir(tf_raw_dir) if os.path.isdir(os.path.join(tf_raw_dir, d))]
        logger.info(f"Found TFBS subdirectories: {subdirs}")
        
        for subdir in sorted(subdirs):
            subdir_raw_path = os.path.join(tf_raw_dir, subdir)
            subdir_processed_path = os.path.join(tf_processed_dir, subdir)
            
            tf_files = ["train.csv", "dev.csv", "test.csv"]
            # Sometimes a split might be missing. We check for train.csv at least
            if os.path.exists(os.path.join(subdir_raw_path, "train.csv")):
                logger.info(f"Processing TFBS subdirectory '{subdir}'...")
                
                dfs_tf = []
                for filename in tf_files:
                    file_path = os.path.join(subdir_raw_path, filename)
                    if os.path.exists(file_path):
                        df_part = pd.read_csv(file_path)
                        logger.info(f"  Loaded {subdir}/{filename}: {len(df_part)} samples")
                        dfs_tf.append(df_part)
                        
                merged_tf_df = pd.concat(dfs_tf, ignore_index=True)
                logger.info(f"Total merged TFBS/{subdir} samples: {len(merged_tf_df)}")
                
                # Preprocess
                cleaned_tf_df, tf_stats = preprocess_dataframe(merged_tf_df)
                
                # Split deterministically
                train_df_tf, val_df_tf, test_df_tf = prevent_leakage_and_split(
                    cleaned_tf_df,
                    val_split=val_split,
                    test_split=test_split,
                    seed=seed
                )
                
                # Save splits
                os.makedirs(subdir_processed_path, exist_ok=True)
                train_df_tf.to_csv(os.path.join(subdir_processed_path, "train.csv"), index=False)
                val_df_tf.to_csv(os.path.join(subdir_processed_path, "val.csv"), index=False)
                test_df_tf.to_csv(os.path.join(subdir_processed_path, "test.csv"), index=False)
                
                with open(os.path.join(subdir_processed_path, "preprocessing_stats.yaml"), "w") as f:
                    yaml.dump(tf_stats, f)
                    
                logger.info(f"Processed TFBS/{subdir} data saved to: {subdir_processed_path}")
    else:
        logger.warning(f"Raw TFBS directory not found at {tf_raw_dir}. Skipping TFBS processing.")

    logger.info("Data preparation completed successfully.")

if __name__ == "__main__":
    main()
