import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class KMeansOutput2 {

    private static final double W_MIN = 0.0, W_MAX = 10_000.0;
    private static final double X_MIN = 20_000.0, X_MAX = 1_000_000.0;
    private static final double Y_MIN = 0.0, Y_MAX = 1.0;
    private static final double Z_MIN = 0.0, Z_MAX = 1.0;

    public static class KMeansMapper extends Mapper<Object, Text, IntWritable, Text> {
        private List<double[]> centroids = new ArrayList<>();
        private IntWritable nearestCentroidId = new IntWritable();
        private Text pointText = new Text();

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            Path[] localFiles = DistributedCache.getLocalCacheFiles(context.getConfiguration());
            if (localFiles == null || localFiles.length == 0) {
                throw new IOException("No seeds file in DistributedCache");
            }
            Path seedsPath = localFiles[0];
            FileSystem localFs = FileSystem.getLocal(context.getConfiguration());
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(localFs.open(seedsPath)))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    line = line.trim();
                    if (line.isEmpty()) continue;
                    String[] parts = line.split(",");
                    centroids.add(new double[]{
                        Double.parseDouble(parts[0]), Double.parseDouble(parts[1]),
                        Double.parseDouble(parts[2]), Double.parseDouble(parts[3])
                    });
                }
            }
        }

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            if (line.startsWith("w,x,y,z") || line.trim().isEmpty()) return;

            String[] parts = line.split(",");
            double[] point = new double[]{
                Double.parseDouble(parts[0]), Double.parseDouble(parts[1]),
                Double.parseDouble(parts[2]), Double.parseDouble(parts[3])
            };

            int nearestId = -1;
            double minDistance = Double.MAX_VALUE;
            for (int i = 0; i < centroids.size(); i++) {
                double[] c = centroids.get(i);
                double distance = Math.sqrt(
                    Math.pow(c[0] - point[0], 2) + Math.pow(c[1] - point[1], 2) +
                    Math.pow(c[2] - point[2], 2) + Math.pow(c[3] - point[3], 2)
                );
                if (distance < minDistance) {
                    minDistance = distance;
                    nearestId = i;
                }
            }

            nearestCentroidId.set(nearestId);
            pointText.set(line + ",1");
            context.write(nearestCentroidId, pointText);
        }
    }

    public static class KMeansCombiner extends Reducer<IntWritable, Text, IntWritable, Text> {
        private Text partialSumText = new Text();

        @Override
        public void reduce(IntWritable key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            double sumW = 0, sumX = 0, sumY = 0, sumZ = 0;
            int count = 0;
            for (Text val : values) {
                String[] parts = val.toString().split(",");
                sumW += Double.parseDouble(parts[0]);
                sumX += Double.parseDouble(parts[1]);
                sumY += Double.parseDouble(parts[2]);
                sumZ += Double.parseDouble(parts[3]);
                count += Integer.parseInt(parts[4]);
            }
            partialSumText.set(sumW + "," + sumX + "," + sumY + "," + sumZ + "," + count);
            context.write(key, partialSumText);
        }
    }

    public static class KMeansReducer extends Reducer<IntWritable, Text, IntWritable, Text> {
        private Text newCentroidText = new Text();

        @Override
        public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            double sumW = 0, sumX = 0, sumY = 0, sumZ = 0;
            int count = 0;
            for (Text val : values) {
                String[] parts = val.toString().split(",");
                sumW += Double.parseDouble(parts[0]);
                sumX += Double.parseDouble(parts[1]);
                sumY += Double.parseDouble(parts[2]);
                sumZ += Double.parseDouble(parts[3]);
                count += Integer.parseInt(parts[4]);
            }
            double newW = sumW / count;
            double newX = sumX / count;
            double newY = sumY / count;
            double newZ = sumZ / count;
            newCentroidText.set(newW + "," + newX + "," + newY + "," + newZ);
            context.write(key, newCentroidText);
        }
    }

    /**
     * Mapper for assignment job: assigns each point to nearest centroid and outputs
     * point with its cluster center.
     */
    public static class AssignMapper extends Mapper<Object, Text, IntWritable, Text> {
        private List<double[]> centroids = new ArrayList<>();
        private IntWritable clusterId = new IntWritable();
        private Text pointWithCentroid = new Text();

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            Path[] localFiles = DistributedCache.getLocalCacheFiles(context.getConfiguration());
            if (localFiles == null || localFiles.length == 0) {
                throw new IOException("No centroids file in DistributedCache");
            }
            Path centroidsPath = localFiles[0];
            FileSystem localFs = FileSystem.getLocal(context.getConfiguration());
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(localFs.open(centroidsPath)))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    line = line.trim();
                    if (line.isEmpty()) continue;
                    String[] parts = line.split(",");
                    centroids.add(new double[]{
                        Double.parseDouble(parts[0]), Double.parseDouble(parts[1]),
                        Double.parseDouble(parts[2]), Double.parseDouble(parts[3])
                    });
                }
            }
        }

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            if (line.startsWith("w,x,y,z") || line.trim().isEmpty()) return;

            String[] parts = line.split(",");
            double[] point = new double[]{
                Double.parseDouble(parts[0]), Double.parseDouble(parts[1]),
                Double.parseDouble(parts[2]), Double.parseDouble(parts[3])
            };

            int nearestId = -1;
            double minDistance = Double.MAX_VALUE;
            for (int i = 0; i < centroids.size(); i++) {
                double[] c = centroids.get(i);
                double distance = Math.sqrt(
                    Math.pow(c[0] - point[0], 2) + Math.pow(c[1] - point[1], 2) +
                    Math.pow(c[2] - point[2], 2) + Math.pow(c[3] - point[3], 2)
                );
                if (distance < minDistance) {
                    minDistance = distance;
                    nearestId = i;
                }
            }

            double[] centroid = centroids.get(nearestId);
            String centroidStr = centroid[0] + "," + centroid[1] + "," + centroid[2] + "," + centroid[3];
            clusterId.set(nearestId);
            pointWithCentroid.set(line + "\t" + centroidStr);
            context.write(clusterId, pointWithCentroid);
        }
    }

    /**
     * Identity reducer for assignment job - passes through point with centroid.
     */
    public static class AssignReducer extends Reducer<IntWritable, Text, IntWritable, Text> {
        @Override
        public void reduce(IntWritable key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            for (Text val : values) {
                context.write(key, val);
            }
        }
    }

    private static double euclideanDistance(double[] a, double[] b) {
        return Math.sqrt(
            Math.pow(a[0] - b[0], 2) + Math.pow(a[1] - b[1], 2) +
            Math.pow(a[2] - b[2], 2) + Math.pow(a[3] - b[3], 2)
        );
    }

    private static double computeCentroidMovement(List<double[]> oldCentroids, Map<Integer, double[]> newCentroids) {
        double totalMovement = 0;
        for (int i = 0; i < oldCentroids.size(); i++) {
            double[] newCentroid = newCentroids.get(i);
            if (newCentroid == null) return Double.MAX_VALUE;
            totalMovement += euclideanDistance(oldCentroids.get(i), newCentroid);
        }
        return totalMovement;
    }

    private static List<double[]> readOldCentroidsFromSeeds(FileSystem fs, Path seedsPath) throws IOException {
        List<double[]> centroids = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(seedsPath)))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                String[] parts = line.split(",");
                centroids.add(new double[]{
                    Double.parseDouble(parts[0]), Double.parseDouble(parts[1]),
                    Double.parseDouble(parts[2]), Double.parseDouble(parts[3])
                });
            }
        }
        return centroids;
    }

    private static Map<Integer, double[]> readNewCentroidsFromReducerOutput(FileSystem fs, Path outputDir) throws IOException {
        Map<Integer, double[]> newCentroids = new HashMap<>();
        org.apache.hadoop.fs.FileStatus[] outputFiles = fs.listStatus(outputDir,
                path -> path.getName().startsWith("part-r-"));
        for (org.apache.hadoop.fs.FileStatus fileStatus : outputFiles) {
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(fileStatus.getPath())))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] parts = line.split("\t");
                    int clusterId = Integer.parseInt(parts[0]);
                    String[] coords = parts[1].split(",");
                    newCentroids.put(clusterId, new double[]{
                        Double.parseDouble(coords[0]), Double.parseDouble(coords[1]),
                        Double.parseDouble(coords[2]), Double.parseDouble(coords[3])
                    });
                }
            }
        }
        return newCentroids;
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 5) {
            System.err.println("Usage: KMeansOutput2 <k> <R> <threshold> <input path> <output path>");
            System.exit(-1);
        }

        int k, R;
        double threshold;
        try {
            k = Integer.parseInt(args[0]);
            R = Integer.parseInt(args[1]);
            threshold = Double.parseDouble(args[2]);
            if (k <= 0 || R <= 0) {
                System.err.println("Error: k and R must be positive integers");
                System.exit(-1);
                return;
            }
            if (threshold < 0) {
                System.err.println("Error: threshold must be non-negative");
                System.exit(-1);
                return;
            }
        } catch (NumberFormatException e) {
            System.err.println("Error: k and R must be positive integers, threshold must be a number");
            System.exit(-1);
            return;
        }

        Configuration conf = new Configuration();
        String inputPath = args[3];
        String outputPath = args[4];

        Random random = new Random();
        File localSeedsFile = File.createTempFile("kmeans_seeds_", ".txt");
        try (PrintWriter writer = new PrintWriter(localSeedsFile)) {
            for (int i = 0; i < k; i++) {
                double w = W_MIN + random.nextDouble() * (W_MAX - W_MIN);
                double x = X_MIN + random.nextDouble() * (X_MAX - X_MIN);
                double y = Y_MIN + random.nextDouble() * (Y_MAX - Y_MIN);
                double z = Z_MIN + random.nextDouble() * (Z_MAX - Z_MIN);
                writer.println(w + "," + x + "," + y + "," + z);
            }
        }

        FileSystem fs = FileSystem.get(conf);
        Path hdfsSeedsPath = new Path("/tmp/kmeans_seeds_" + System.currentTimeMillis() + ".txt");
        fs.copyFromLocalFile(new Path(localSeedsFile.getAbsolutePath()), hdfsSeedsPath);
        localSeedsFile.delete();

        boolean success = true;
        int lastIter = 0;

        for (int iter = 0; iter < R; iter++) {
            lastIter = iter;
            String iterOutputPath = outputPath + "/iter_" + iter;

            Job job = Job.getInstance(conf, "4D K-Means Iteration " + iter);
            job.setJarByClass(KMeansOutput2.class);
            job.setMapperClass(KMeansMapper.class);
            job.setCombinerClass(KMeansCombiner.class);
            job.setReducerClass(KMeansReducer.class);
            job.setOutputKeyClass(IntWritable.class);
            job.setOutputValueClass(Text.class);
            job.addCacheFile(hdfsSeedsPath.toUri());
            FileInputFormat.addInputPath(job, new Path(inputPath));
            FileOutputFormat.setOutputPath(job, new Path(iterOutputPath));

            success = job.waitForCompletion(true);
            if (!success) {
                System.err.println("Job failed at iteration " + iter);
                System.exit(1);
            }

            if (iter < R - 1) {
                List<double[]> oldCentroids = readOldCentroidsFromSeeds(fs, hdfsSeedsPath);
                Map<Integer, double[]> newCentroids = readNewCentroidsFromReducerOutput(fs, new Path(iterOutputPath));
                double totalMovement = computeCentroidMovement(oldCentroids, newCentroids);

                if (totalMovement < threshold) break;

                Path newSeedsPath = new Path("/tmp/kmeans_seeds_iter_" + (iter + 1) + ".txt");
                try (PrintWriter writer = new PrintWriter(new File(newSeedsPath.getName()))) {
                    for (int i = 0; i < k; i++) {
                        double[] c = newCentroids.get(i);
                        if (c != null) {
                            writer.println(c[0] + "," + c[1] + "," + c[2] + "," + c[3]);
                        } else {
                            double[] old = oldCentroids.get(i);
                            writer.println(old[0] + "," + old[1] + "," + old[2] + "," + old[3]);
                        }
                    }
                }
                fs.copyFromLocalFile(new Path(newSeedsPath.getName()), newSeedsPath);
                new File(newSeedsPath.getName()).delete();
                hdfsSeedsPath = newSeedsPath;
            }
        }

        // Build final centroids file for assignment job (clusterId order: 0,1,...,k-1)
        Map<Integer, double[]> finalCentroids = readNewCentroidsFromReducerOutput(fs, new Path(outputPath + "/iter_" + lastIter));
        List<double[]> fallbackCentroids = readOldCentroidsFromSeeds(fs, hdfsSeedsPath);
        Path finalCentroidsPath = new Path("/tmp/kmeans_final_centroids_" + System.currentTimeMillis() + ".txt");
        File localCentroidsFile = File.createTempFile("kmeans_centroids_", ".txt");
        try (PrintWriter writer = new PrintWriter(localCentroidsFile)) {
            for (int i = 0; i < k; i++) {
                double[] c = finalCentroids.get(i);
                if (c == null && i < fallbackCentroids.size()) {
                    c = fallbackCentroids.get(i);
                }
                if (c != null) {
                    writer.println(c[0] + "," + c[1] + "," + c[2] + "," + c[3]);
                }
            }
        }
        fs.copyFromLocalFile(new Path(localCentroidsFile.getAbsolutePath()), finalCentroidsPath);
        localCentroidsFile.delete();

        // Assignment job: output each point with its cluster center
        Path clusteredOutputPath = new Path(outputPath + "/clustered");
        Job assignJob = Job.getInstance(conf, "K-Means Assign Points to Clusters");
        assignJob.setJarByClass(KMeansOutput2.class);
        assignJob.setMapperClass(AssignMapper.class);
        assignJob.setReducerClass(AssignReducer.class);
        assignJob.setOutputKeyClass(IntWritable.class);
        assignJob.setOutputValueClass(Text.class);
        assignJob.addCacheFile(finalCentroidsPath.toUri());
        FileInputFormat.addInputPath(assignJob, new Path(inputPath));
        FileOutputFormat.setOutputPath(assignJob, clusteredOutputPath);

        success = assignJob.waitForCompletion(true);

        System.exit(success ? 0 : 1);
    }
}
