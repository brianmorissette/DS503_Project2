import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;


// Originally create a custom tuple class to store the data in a structured way
//But now it's optimized for the reducer

public class TaskB {

    private static class UserRecord implements Comparable<UserRecord> {
        int userId;
        String nickname;
        String jobTitle;
        int count;

        public UserRecord(int userId, String nickname, String jobTitle, int count) {
            this.userId = userId;
            this.nickname = nickname;
            this.jobTitle = jobTitle;
            this.count = count;
        }

        @Override
        public int compareTo(UserRecord other) {
            return Integer.compare(this.count, other.count);
        }
    }

     //Mapper class for Task B
    public static class TaskBMapper extends Mapper<Object, Text, IntWritable, IntWritable> {

        
        private final static IntWritable one = new IntWritable(1);
        private IntWritable id = new IntWritable();
        //This is the ID column for accessed page
        private int targetColumn_Activity = 2; 

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            
            //take the record it and split it by column
            String line = value.toString();
            String[] attr = line.split(",");

            //Make sure we skip the header
            if(attr[targetColumn_Activity].equals("WhatPage")) return;

            //set the key to the page
            id.set(Integer.parseInt(attr[targetColumn_Activity]));

            //This is where we set the key value pair
            context.write(id, one);
        }   
    }

    //Reducer class for Task B
    public static class TaskBReducer extends Reducer<IntWritable,IntWritable,IntWritable,Text> {

        //HashMap to store user ID and nickname
        private Map<Integer, String[]> users = new HashMap<>();

        private PriorityQueue<UserRecord> topUsers = new PriorityQueue<>(10);

        //Reducer side join
        @Override
        protected void setup(Context context) throws IOException, InterruptedException {

            //Get the cached file
            java.net.URI[] cacheFiles = context.getCacheFiles();

            if (cacheFiles != null && cacheFiles.length > 0) {

                // Extract the local filename from the URI
                String localFileName = new Path(cacheFiles[0].getPath()).getName(); 

                //Read the file
                try (BufferedReader reader = new BufferedReader(new FileReader(localFileName))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        String[] parts = line.split(",");

                        //Skip header
                        if (parts[0].equals("id"))
                            continue;
                        
                        int userId = Integer.parseInt(parts[0]);
                        String nickname = parts[1];
                        String jobTitle = parts[2];
                        users.put(userId, new String[]{nickname, jobTitle});
                    }
                }
            }
        }


        public void reduce(IntWritable key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            
            String nickName = "";
            String jobTitle = "";
            int sum = 0;

            //Iterate through the values
            for (IntWritable val : values) {
                sum += val.get();
            }

            //Get the nickname and job title from the hashmap
            nickName = users.get(key.get())[0];
            jobTitle = users.get(key.get())[1];

            //Create a new record
            UserRecord record = new UserRecord(key.get(), nickName, jobTitle, sum);

            //Check if the queue is full
            if (topUsers.size() < 10) {
                topUsers.add(record);
            } else if (record.compareTo(topUsers.peek()) > 0) {

                //If the current record is greater than the smallest record in the queue
                //Remove the smallest record and add the current record
                topUsers.poll();
                topUsers.add(record);
            }
        }

        //Once all of the data has been aggregated, we can output only the top 10
        //all of which will be in the priority queue
        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            while (!topUsers.isEmpty()) {
                UserRecord record = topUsers.poll();
                context.write(
                    new IntWritable(record.userId),
                    new Text(record.nickname + "\t" + record.jobTitle + "\t" + record.count)
                );
            }
        }
    }

    public static class TaskBCombiner extends Reducer<IntWritable,IntWritable,IntWritable,IntWritable> {

        public void reduce(IntWritable key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            context.write(key, new IntWritable(sum));
        }
    }   

    //Main method
    public static void main(String[] args) throws Exception {

        //Create a new job
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Task B");
        
        //Set the job class
        job.setJarByClass(TaskB.class);
        
        //Set the mapper class
        job.setMapperClass(TaskBMapper.class);

        //set the combiner class
        job.setCombinerClass(TaskBCombiner.class);
        
        //Set the reducer class
        job.setReducerClass(TaskBReducer.class);

        //Set the number of reduce tasks
        //This will allow us to get only the top 10
        job.setNumReduceTasks(1);

        //Add the cache file
        //This is where we add the accounts.csv file to the reducer
        job.addCacheFile(new Path("/assignment1/accounts.csv").toUri());

        //Set the output key and value classes
        //This is where you set the data types for the key and value output
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Text.class);

        //Set the map output key and value classes
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(IntWritable.class);
        
        //Set the input and output paths
        //This is where you set the input and output paths
        FileInputFormat.addInputPath(job, new Path("/assignment1/activity.csv"));
        FileOutputFormat.setOutputPath(job, new Path(args[0]));
        
        //Run the job
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }   

}
