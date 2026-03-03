-- Load dataset
accounts = LOAD '/assignment1/accounts.csv' 
    USING PigStorage(',') 
    AS (id:int, name:chararray, jobtitle:chararray, region:int, hobby:chararray);

follows = LOAD '/assignment1/follows.csv' 
    USING PigStorage(',') 
    AS (colrel:int, id1:int, id2:int, date:int, desc:chararray);

get_follows = GROUP follows BY id2;

follow_count = FOREACH get_follows GENERATE group AS id2, COUNT(follows) AS count;

-- select by hobby
join_accounts = JOIN accounts BY id LEFT OUTER, follow_count BY id2;

result = FOREACH join_accounts GENERATE
    accounts::id,
    accounts::name,
    (follow_count::count IS NOT NULL ? follow_count::count : 0) AS num_followers;

-- Store the result
STORE result INTO '/assignment2/taskD' USING PigStorage(',');