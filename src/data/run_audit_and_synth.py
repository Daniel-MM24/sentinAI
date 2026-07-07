import os
import sys
import subprocess
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_subprocess(script_path: str):
    logger.info(f"Executing decoupled script: {script_path}")
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Script {script_path} failed with exit code {result.returncode}")
        logger.error(f"STDOUT:\n{result.stdout}")
        logger.error(f"STDERR:\n{result.stderr}")
        sys.exit(result.returncode)
    else:
        logger.info(f"Script {script_path} completed successfully.")
        logger.info(f"STDOUT:\n{result.stdout}")
        # stderr might contain warnings
        if result.stderr:
            logger.warning(f"STDERR:\n{result.stderr}")

def main():
    base_dir = Path(__file__).parent
    
    scripts = [
        base_dir / "run_bronze.py",
        base_dir / "run_silver.py",
        base_dir / "run_gold.py"
    ]
    
    for script in scripts:
        if not script.exists():
            logger.error(f"Required script not found: {script}")
            sys.exit(1)
            
        run_subprocess(str(script))

if __name__ == "__main__":
    main()
