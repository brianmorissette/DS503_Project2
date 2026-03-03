-- Load dataset
accounts = LOAD '/assignment1/accounts.csv' 
    USING PigStorage(',') 
    AS (id:int, name:chararray, jobtitle:chararray, region:int, hobby:chararray);

-- select by hobby
check_hobby = FILTER accounts BY hobby == 'Playing the guitar';

result = FOREACH check_hobby GENERATE
    id,
    name,
    jobtitle,
    hobby;

-- Store the result
STORE result INTO '/assignment2/taskC' USING PigStorage(',');