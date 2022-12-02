pyqc: A Python package for doing simple QA/QC checks on weather station
data.
================

`pyqc` is a python python package for performing QA/QC checks on weather
station data that are stored in a Pandas DataFrame. The package
currently implements [three tests suggested by the Oklahoma
Mesonet](https://cig.mesonet.org/staff/shafer/Mesonet_QA_final_fixed.pdf)
(with more planned in the future):

1.  Range Test: For a given measurement, an upper and lower threshold of
    reasonable measurement values are provided. If the measurement falls
    outside of those values, a QA flag is raised.
2.  Step Test: For two consecutive measurements of a given variable, the
    difference between the two values is compared to a maximum allowable
    “step” size. If the difference is larger than the step size, a QA
    flag is raised.
3.  Persistence Test: The standard deviation of all values of a given
    variable over the course of a day is calculated. If the standard
    deviation is smaller than a specified threshold value, a QA flag is
    raised for all values that day.

The `pyqc` package assumes that observations are in a `long` data format
(i.e., each row in a table corresponds to one observation. For example,
here is some long formatted data from the Montana Mesonet where each row
corresponds to one observation for one element (variable):

``` python
import pandas as pd

# Read two days of data from a Montana Mesonet station. 
observations = pd.read_csv(
    "https://mesonet.climate.umt.edu/api/v2/observations?stations=aceabsar&start_time=2022-10-01&end_time=2022-10-03&wide=False&type=csv&level=0"
)
print(observations.head())
```

        station                   datetime        element   value units
    0  aceabsar  2022-10-01 00:00:00-06:00  air_temp_0200  47.984  degF
    1  aceabsar  2022-10-01 00:05:00-06:00  air_temp_0200  49.352  degF
    2  aceabsar  2022-10-01 00:10:00-06:00  air_temp_0200  49.676  degF
    3  aceabsar  2022-10-01 00:15:00-06:00  air_temp_0200  49.064  degF
    4  aceabsar  2022-10-01 00:20:00-06:00  air_temp_0200  48.992  degF

In order to perform the QA/QC checks in `pyqc`, you must have a
DataFrame specifying the QA thresholds for each element:

``` python
elements = pd.read_csv("./test/elements.csv")
print(elements.head())
```

        station        element  ...          model serial_number
    0  aceabsar  air_temp_0200  ...        HMP155E      U0940850
    1  aceabsar  air_temp_0200  ...        HMP155E      U0940660
    2  aceabsar  air_temp_0200  ...     HygroVUE10         E1530
    3  aceabsar             bp  ...      278 CS100       7599141
    4  aceabsar            ppt  ...  Pluvio2 L 400    L100452357

    [5 rows x 13 columns]

For each element, the above DataFrame provides a wide variety of
metadata, including the measurement type and the manufacturer of the
instrument. However, the key information in the data frame are the
`range_min`, `range_max`, `step_size` and `persistence_delta` columns.
These columns contain the necessary data to perform all the QA/QC tests.
Now, using our `observations` and `elements` DataFrames, we can perform
the three QA/QC tests outlined above:

``` python
import pyqc.checks as ck
from pyqc.columns import Columns
from pyqc.process import check_observations

# Columns is a class that maps the columns in a the dataframes to something interpretable by
# the pyqc functions. If your columns have different names than those outlined above, use 
# the Columns class to provide a different mapping (see help(Columns)). 
columns = Columns()

# You can specify which checks to run. For example, if you only want to run these checks 
# in real time, you can omit the ck.check_variance_pd test so a whole day of data isn't
# needed to run the check. 
checks = [ck.check_range_pd, ck.check_step_pd, ck.check_variance_pd]

checked = check_observations(observations, elements, columns, *checks)
print(checked.head())
```

        station                   datetime  ... qa_step  qa_delta
    0  aceabsar  2022-10-01 00:00:00-06:00  ...       0         0
    1  aceabsar  2022-10-01 00:00:00-06:00  ...       0         0
    2  aceabsar  2022-10-01 00:00:00-06:00  ...       0         0
    3  aceabsar  2022-10-01 00:00:00-06:00  ...       0         0
    4  aceabsar  2022-10-01 00:00:00-06:00  ...       0         0

    [5 rows x 7 columns]

In the resulting DataFrame, a new column prefixed with `qa_` is added
for each QA check that was performed. For each check, a value of 0
indicates that the data are of good quality and that nothing appears to
be wrong, and a value of 1 indicates that the check failed.
