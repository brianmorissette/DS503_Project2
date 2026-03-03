import java.io.IOException;
import java.util.HashSet;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;


//Currently WRONG!!!!
public class TaskE {


     //Mapper class for Task E
    public static class TaskEMapper extends Mapper<Object, Text, IntWritable, IntWritable> {
        
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            
            //take the record it and split it by column
            String line = value.toString();
            String[] attr = line.split(",");

            if(attr[0].equals("ActionId")) return;

            //Write to the reducer the id of the account that did the action, and what page
            context.write(new IntWritable(Integer.parseInt(attr[1])), new IntWritable(Integer.parseInt(attr[2])));
        }
    }

    //Reducer class for Task B
    public static class TaskEReducer extends Reducer<IntWritable,IntWritable,IntWritable,Text> { 

        public void reduce(IntWritable key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            
            //Have an array of seen pageID's, if its in the array skip it in unique but still count as an access
            HashSet<Integer> seenPages = new HashSet<>();
            
            int actions = 0;
            int uniquePages = 0;

            //Iterate through the values
            for (IntWritable val : values) {

                actions++;
                int pageId = val.get();

                
                // Only increment uniquePages if not seen before
                if (!seenPages.contains(pageId)) {
                    seenPages.add(pageId);
                    uniquePages++;
                }
            }

            //This is where we write the key value pair
            context.write(key, new Text(actions + ", " + uniquePages));   
        }
    }

    //Main method
    public static void main(String[] args) throws Exception {

        //Create a new job
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Task E");
        
        //Set the job class
        job.setJarByClass(TaskE.class);
        
        //Set the mapper class
        job.setMapperClass(TaskEMapper.class);
        
        //Set the reducer class
        job.setReducerClass(TaskEReducer.class);

        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(IntWritable.class);

        //Set the output key and value classes
        //This is where you set the data types for the key and value output
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Text.class);
        
        //Set the input and output paths
        //This is where you set the input and output paths
        FileInputFormat.addInputPath(job, new Path("/assignment1/activity.csv"));
        FileOutputFormat.setOutputPath(job, new Path(args[0]));
        
        //Run the job
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }   


}
