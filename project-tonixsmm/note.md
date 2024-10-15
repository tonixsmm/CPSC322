1. Take a look on 1.3.6. Didn't decide if we will remove it or not.
2. Also, didn't touch on zip_code and state
3. deling_2yrs didn't touch
4. earliest_cr_line. Needs additional cleaning to make sure month-year is compliant
    Needs to take care all of the day attributes
5. Merge ZHVI on issue_d. Calculate mean/median(?) house value until the last payment date is made
    Should we factor in inflation?

At roughly 780,000 instances and 27 columns, the dataset is too large and takes too long to run. Based on the