import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def save_field_npy(
    field: np.ndarray,
    filepath: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save a complex field to NPY format.
    
    Parameters
    ----------
    field : np.ndarray
        Complex field array
    filepath : str or Path
        Output file path
    metadata : dict, optional
        Metadata to save alongside (as separate JSON file)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    np.save(filepath, field)
    logger.info(f"Saved field to {filepath}")
    
    if metadata is not None:
        import json
        meta_path = filepath.with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {meta_path}")


def load_field_npy(filepath: Union[str, Path]) -> np.ndarray:
    """
    Load a complex field from NPY format.
    
    Parameters
    ----------
    filepath : str or Path
        Input file path
        
    Returns
    -------
    np.ndarray
        Loaded field
    """
    field = np.load(filepath)
    logger.info(f"Loaded field from {filepath}")
    return field


def save_field_hdf5(
    field: np.ndarray,
    filepath: Union[str, Path],
    dataset_name: str = "field",
    metadata: Optional[Dict[str, Any]] = None,
    compression: str = "gzip"
):
    """
    Save a complex field to HDF5 format.
    
    Parameters
    ----------
    field : np.ndarray
        Complex field array
    filepath : str or Path
        Output file path
    dataset_name : str, optional
        Name of dataset in HDF5 file
    metadata : dict, optional
        Metadata to store as attributes
    compression : str, optional
        Compression method ('gzip', 'lzf', None)
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required for HDF5 support. Install with: pip install h5py")
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(filepath, 'w') as f:
        dset = f.create_dataset(
            dataset_name,
            data=field,
            compression=compression
        )
        
        if metadata is not None:
            for key, value in metadata.items():
                dset.attrs[key] = value
    
    logger.info(f"Saved field to {filepath} (HDF5)")


def load_field_hdf5(
    filepath: Union[str, Path],
    dataset_name: str = "field"
) -> np.ndarray:
    """
    Load a complex field from HDF5 format.
    
    Parameters
    ----------
    filepath : str or Path
        Input file path
    dataset_name : str, optional
        Name of dataset to load
        
    Returns
    -------
    np.ndarray
        Loaded field
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required for HDF5 support. Install with: pip install h5py")
    
    with h5py.File(filepath, 'r') as f:
        field = f[dataset_name][:]
    
    logger.info(f"Loaded field from {filepath} (HDF5)")
    return field