-- Load dataset
accounts = LOAD '/assignment1/accounts.csv' 
    USING PigStorage(',') 
    AS (id:int, name:chararray, jobtitle:chararray, region:int, hobby:chararray);

-- Group by hobby
hobby_group = GROUP accounts BY hobby;

-- Count hobbies
hobby_count = FOREACH hobby_group GENERATE group AS hobby, COUNT(accounts) AS count;

-- Order by count descending
hobby_count_sorted = ORDER hobby_count BY count DESC;

-- Store the result
STORE hobby_count_sorted INTO '/assignment2/taskA' USING PigStorage(',');