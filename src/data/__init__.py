"""
src/data package for satellite data ingestion, calibration, spatial cropping, and sequence generation.
"""
from src.data.goes_reader import (
    GOESProjection,
    read_goes_single_file,
    read_multichannel_timestamp,
    build_timestamp_catalog,
    compute_file_sha256,
    audit_catalog_duplicates,
)
from src.data.spatial_crop import (
    GeographicCropper,
    extract_patches,
)
from src.data.sequence_generator import (
    TemporalSequenceGenerator,
)
from src.data.insat_reader import (
    IR_CHANNELS,
    calibrate_counts_to_kelvin,
    read_insat_channel,
    read_insat_multichannel_frame,
    INSATRegionalCropper,
    combine_preprocessing_masks,
)
