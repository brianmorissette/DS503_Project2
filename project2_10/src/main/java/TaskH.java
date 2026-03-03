
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;

import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;


public class TaskH {
    
    //Mapper class for Task H
    public static class TaskHMapper extends Mapper<Object, Text, Text, Text> {

        
       
        
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            
            //take the record it and split it by column
            String line = value.toString();
            String[] attr = line.split(",");

            if(attr[0].equals("Colrel")) return;

            //Write to 2 reducers with either real or fake follows
            context.write(new Text(attr[1] + ',' + attr[2]), new Text("Real"));
            context.write(new Text(attr[2] + ',' + attr[1]), new Text("Fake"));
            
        }
    }

    //Reducer class for Task H
    public static class TaskHReducer extends Reducer<Text,Text,IntWritable,Text> {

        //HashMap to store user ID and nickname
        private Map<Integer, String[]> users = new HashMap<>();

        //Set to store users that have already been emitted
        Set<Integer> emittedUsers = new HashSet<>();


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
                        String region = parts[3];
                        users.put(userId, new String[]{nickname, region});
                    }
                }
            }
        }

        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            
            //Split the key into two parts one for each account
            String[] parts = key.toString().split(",");
            int userId = Integer.parseInt(parts[0]);
            String nickname = users.get(userId)[0];

            //Get region codes of both users
            String region1 = users.get(Integer.parseInt(parts[0]))[1];
            String region2 = users.get(Integer.parseInt(parts[1]))[1];

            //Check if the follow is real or fake
            String type = "";
            int num = 0;

            //Iterate through the values
            for (Text val : values) {
                num++;
                type = val.toString();
            }

            //If there are more than 1 value, it means the follow is both real and fake
            if (num > 1) return;

            //If the follow is real and the users are from the same region, write to the output
            if (type.equals("Real") && region1.equals(region2)){
                if (!emittedUsers.contains(userId)) {
                    context.write(new IntWritable(userId), new Text(nickname));
                    emittedUsers.add(userId);
                }  

            }

        }
    }

    //Main method
    public static void main(String[] args) throws Exception {

        //Create a new job
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Task H");
        
        //Set the job class
        job.setJarByClass(TaskH.class);
        
        //Set the mapper class
        job.setMapperClass(TaskHMapper.class);
        
        //Set the map output key and value classes
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        
        //Set the reducer class
        job.setReducerClass(TaskHReducer.class);

        //Set the output key and value classes
        //This is where you set the data types for the key and value output
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Text.class);
        
        //Cache the accounts file for the reducer side join
        job.addCacheFile(new Path("/assignment1/accounts.csv").toUri());
        
        //Set the input and output paths
        //This is where you set the input and output paths
        FileInputFormat.addInputPath(job, new Path("/assignment1/follows.csv"));
        FileOutputFormat.setOutputPath(job, new Path(args[0]));
        
        //Run the job
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }   

}
