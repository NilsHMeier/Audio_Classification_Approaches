import numpy as np
import pandas as pd
import datetime as dt


def aggregate_predictions(predictions: np.ndarray, step_size: float, aggregation_period: float,
                          start_time: dt.datetime) -> pd.DataFrame:
    # Create dataframe of predictions with timestamps
    pred_df = pd.DataFrame(data={'timestamp': [start_time + dt.timedelta(seconds=step_size * i) for i in
                                               range(len(predictions))],
                                 'prediction': predictions})

    # Create dataframe for aggregated predictions starting at begin of aggregation period
    start_time = start_time.replace(second=start_time.second - int(start_time.second % aggregation_period))
    timestamps = [start_time + dt.timedelta(seconds=aggregation_period * i) for i in
                  range(0, int(len(predictions) * step_size / aggregation_period + 1))]
    aggregated_df = pd.DataFrame(data={'timestamp': timestamps, 'cars': np.nan})

    # Aggregate predictions
    for i in aggregated_df.index:
        # Select relevant rows from predictions
        start_timestamp = aggregated_df.at[i, 'timestamp']
        end_timestamp = aggregated_df.at[i, 'timestamp'] + dt.timedelta(seconds=aggregation_period)
        relevant_rows = pred_df[np.logical_and(pred_df['timestamp'] >= start_timestamp,
                                               pred_df['timestamp'] < end_timestamp)]
        aggregated_df.at[i, 'cars'] = np.sum(relevant_rows['prediction'])

    # Return aggregated predictions
    return aggregated_df
