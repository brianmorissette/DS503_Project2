-- Load dataset
activity = LOAD '/assignment1/activity.csv' 
    USING PigStorage(',') 
    AS (actionid:int, userid:int, page:int, actiontype:chararray, actiontime:int);

group_activity = GROUP activity BY userid;

activity_count = FOREACH group_activity {
    distinct_pages = DISTINCT activity.page;
    GENERATE group AS userid, COUNT(activity) AS count, COUNT(distinct_pages) AS distinct_count;
};

result = FOREACH activity_count GENERATE
    userid,
    count,
    distinct_count;

-- Store the result
STORE result INTO '/assignment2/taskE' USING PigStorage(',');