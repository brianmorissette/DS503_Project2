-- Load dataset
accounts = LOAD '/assignment1/accounts.csv' 
    USING PigStorage(',') 
    AS (id:int, name:chararray, jobtitle:chararray, region:int, hobby:chararray);

activity = LOAD '/assignment1/activity.csv'
    USING PigStorage(',')
    AS (actionid:int, userid:int, page:int, actiontype:chararray, actiontime:chararray);

-- Group by page
page_group = GROUP activity BY page;

-- Count page access
access_count = FOREACH page_group GENERATE group AS page, COUNT(activity) AS count;

-- Join with accounts
join_page = Join access_count by page, accounts by id;

result = FOREACH join_page GENERATE
    access_count::page,
    access_count::count,
    accounts::name,
    accounts::jobtitle;

-- Order by count descending
result = ORDER result BY access_count::count DESC;

top_10 = LIMIT result 10;

-- Store the result
STORE top_10 INTO '/assignment2/taskB' USING PigStorage(',');

--note that i put the order too early and it slowed it down a bunch