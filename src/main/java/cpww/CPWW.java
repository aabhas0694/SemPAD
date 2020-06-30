package cpww;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import cpww.utils.PatternMatchIndices;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;

import java.io.*;
import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.logging.FileHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static cpww.utils.Util.*;

public class CPWW {
    private static String data_name;
    private static String inputFolder;
    private static String outputFolder;
    private static int noOfLines;
    private static int minimumSupport;
    private static int maxLength;
    private static StanfordCoreNLP pipeline;
    private static boolean load_sentenceBreakdownData;
    private static boolean load_metapatternData;
    private static int noOfPushUps;
    private static boolean includeContext;
    private static int startBatch;
    private static int batchSize;

    private static String[] nerTypes;
    private static List<String> stopWords;
    private static int noOfBatches = 0;
    private static Map<String, String> entityDictionary = new HashMap<>();
    private static List<MetaPattern> frequentPatterns = new ArrayList<>();
    private static Map<String, List<MetaPattern>> singlePattern = new HashMap<>();
    private static Map<String, List<MetaPattern>> allPattern = new HashMap<>();
    private static Map<String, List<MetaPattern>> multiPattern = new HashMap<>();
    private static Map<Integer, List<String>> allPattern_Index = new HashMap<>();
    private static Map<String, Set<Integer>> allPattern_reverseIndex = new HashMap<>();
    private static ForkJoinPool forkJoinPool = null;

    private static Logger logger = Logger.getLogger(CPWW.class.getName()) ;


    public CPWW()  {
    }

    private static void initialization() throws Exception {
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,parse,depparse");
        pipeline = new StanfordCoreNLP(props);

        InputStream input = new FileInputStream("config.properties");
        Properties prop = new Properties();
        prop.load(input);
        if (!prop.containsKey("data_name") || prop.getProperty("data_name").equals("")) {
            throw new IOException("Parameter 'data_name' <does not exist/ is empty> in config.properties");
        }
        setOptionalParameterDefaults(prop);
        data_name = prop.getProperty("data_name");
        inputFolder = folderNameConsistency(prop.getProperty("inputFolder"));
        outputFolder = folderNameConsistency(prop.getProperty("outputFolder"));
        noOfLines = Integer.parseInt(prop.getProperty("noOfLines"));
        minimumSupport = Integer.parseInt(prop.getProperty("minimumSupport"));
        maxLength = Integer.parseInt(prop.getProperty("maxLength"));
        nerTypes = readList(new FileReader(prop.getProperty("nerTypes"))).toArray(new String[0]);
        stopWords = readList(new FileReader(prop.getProperty("stopWordsFile")));
        load_sentenceBreakdownData = Boolean.parseBoolean(prop.getProperty("load_sentenceBreakdownData"));
        load_metapatternData = Boolean.parseBoolean(prop.getProperty("load_metapatternData"));
        noOfPushUps = Integer.parseInt(prop.getProperty("noOfPushUps"));
        includeContext = Boolean.parseBoolean(prop.getProperty("includeContext"));
        startBatch = Integer.parseInt(prop.getProperty("startFromBatch"));
        batchSize = Integer.parseInt(prop.getProperty("batchSize"));
        int threadCount = Integer.parseInt(prop.getProperty("threadCount"));

        if (!new File(inputFolder).exists()) throw new IOException("Input Folder not found.");
        boolean metadata_exists = new File(inputFolder + "dict.json").exists();
        boolean annotatedDataExists = new File(inputFolder + "annotated.txt").exists();
        if (!metadata_exists || !annotatedDataExists) {
            throw new IOException("Required Input Files not found.");
        }
        if (load_sentenceBreakdownData) {
            String directory = inputFolder + "ProcessedInput/" + batchSize + "/";
            if (!new File(directory).exists()) throw new IOException("Sentence breakdown data does not exist for given batch size.");
            noOfBatches = countSavedBatches(directory, noOfLines, startBatch);
            if (noOfBatches == 0) throw new IOException("Sentence Breakdown Data does not exist.");
        }
        makingDir(outputFolder);
        if (load_metapatternData) {
            String suffix = "." + noOfLines + "_" + minimumSupport + ".txt";
            boolean patternDataExists = new File(outputFolder + "allPatterns" + suffix).exists();
            if (!patternDataExists) {
                throw new IOException("Required Pattern Files not found.");
            }
        }
        makingDir(inputFolder + "ProcessedInput");
        makingDir(inputFolder + "ProcessedInput/" + batchSize);
        String logFileSuffix = "logFile." + noOfLines + "_" + minimumSupport;
        FileHandler logFile = new FileHandler(outputFolder + logFileSuffix + ".txt");
        logFile.setFormatter(new SimpleFormatter());
        logger.addHandler(logFile);
        try {
            forkJoinPool = new ForkJoinPool(threadCount);
        } catch (Exception e) {
            throw new Exception(e);
        }
    }

    private static void frequent_pattern_mining(int iteration) throws Exception {
        logger.log(Level.INFO, "STARTING: Frequent Pattern Mining - Iteration " + iteration);
        frequentPatterns.clear();

        List<SubSentWords> tokens = new ArrayList<>();
        Map<String, List<Integer>> dict_token = new HashMap<>();
        Map<String, List<Integer>> valid_pattern = new HashMap<>();
        List<SentenceProcessor> sentenceCollector;
        int tokenNumber = 0;
        for (int batchNo = startBatch; batchNo < noOfBatches; batchNo++) {
            sentenceCollector = loadSentenceBreakdown(batchNo, iteration == 1 ? -1 : iteration);
            for (SentenceProcessor sentence : sentenceCollector) {
                Map<SubSentWords, List<SubSentWords>> sentBreak = sentence.getSentenceBreakdown();
                for (SubSentWords key : sentBreak.keySet()) {
                    List<SubSentWords> list = sentBreak.get(key);
                    for (SubSentWords word : list) {
                        tokens.add(word);
                        updateMapCount(dict_token, word.getLemma(), tokenNumber++);
                        updateMapCount(valid_pattern, word.getLemma(), 1);
                    }
                    tokens.add(new SubSentWords("$", "$", "", -1));
                    tokenNumber++;
                }
                for (SubSentWords key : sentence.getPushedUpSentences().keySet()) {
                    for (SubSentWords word : sentence.getPushedUpSentences().get(key)) {
                        tokens.add(word);
                        updateMapCount(dict_token, word.getLemma(), tokenNumber++);
                        updateMapCount(valid_pattern, word.getLemma(), 1);
                    }
                    tokens.add(new SubSentWords("$", "$", "", -1));
                    tokenNumber++;
                }
            }
            sentenceCollector.clear();
        }

        int patternLength = 1;
        while(dict_token.size() > 1) {
            if (patternLength > maxLength) break;
            patternLength++;
            Map<String, List<Integer>> newdict_token = new HashMap<>();
            Map<String, List<Integer>> newvalid_pattern = new HashMap<>();
            for (String pattern: dict_token.keySet()){
                List<Integer> positions = dict_token.get(pattern);
                List<Integer> valid = valid_pattern.get(pattern);
                int frequency = positions.size();
                if (frequency >= minimumSupport){
                    if (patternLength > 2 && sum(valid) >= minimumSupport){
                        MetaPattern metaPattern = new MetaPattern(pattern, nerTypes, stopWords, frequency);
                        if (metaPattern.isValid()) {
                            frequentPatterns.add(metaPattern);
                        }
                    }

                    for (Integer i : positions){
                        if (i + 1 < tokenNumber){
                            SubSentWords nextToken = tokens.get(i + 1);
                            if (nextToken.getLemma().equals("$")) continue;
                            String newPattern = pattern + " " + nextToken.getLemma();
                            int new_valid = 1;

                            List<String> pattern_tree = new ArrayList<>();
                            for (int j = i - patternLength + 2; j < i + 2; j++){
                                pattern_tree.add(tokens.get(j).getTrimmedEncoding());
                            }
                            String pattern_root = min_characters(pattern_tree);
                            for (String t : pattern_tree) {
                                if (!pattern_tree.contains(t.substring(0,t.length()-1)) && !t.equals(pattern_root)){
                                    new_valid = 0;
                                    break;
                                }
                            }
                            updateMapCount(newdict_token, newPattern, i + 1);
                            updateMapCount(newvalid_pattern, newPattern, new_valid);
                        }
                    }
                }
            }
            dict_token = new HashMap<>(newdict_token);
            valid_pattern = new HashMap<>(newvalid_pattern);
        }

        logger.log(Level.INFO, "COMPLETED: Frequent Pattern Mining - Iteration " + iteration);
        pattern_classification();
    }

    private static void pattern_classification() {
        clearPatterns();
        int allPatternCount = 0;

        for (MetaPattern metaPattern : frequentPatterns.stream().sorted(
                Comparator.comparingInt(MetaPattern::getFrequency).reversed()).collect(Collectors.toList())) {
            String main_pattern = metaPattern.getClippedMetaPattern();
            String[] splitPattern = main_pattern.split(" ");
            List<MetaPattern> temp;
            if (metaPattern.getNerCount() == 1) {
                temp = singlePattern.getOrDefault(main_pattern, new ArrayList<>());
                temp.add(metaPattern);
                singlePattern.put(main_pattern, temp);
            } else if (metaPattern.getNerCount() > 1) {
                temp = multiPattern.getOrDefault(main_pattern, new ArrayList<>());
                temp.add(metaPattern);
                multiPattern.put(main_pattern, temp);
            }
            if (metaPattern.getNerCount() >= 1) {
                temp = allPattern.getOrDefault(main_pattern, new ArrayList<>());
                temp.add(metaPattern);
                allPattern.put(main_pattern, temp);
                allPatternCount++;
                allPattern_Index.put(allPatternCount,new ArrayList<>(Arrays.asList(splitPattern)));
                for (String word: splitPattern){
                    Set<Integer> indices = new HashSet<>(allPattern_reverseIndex.getOrDefault(word, new HashSet<>()));
                    indices.add(allPatternCount);
                    allPattern_reverseIndex.put(word,indices);
                }
            }
        }
    }

    private static void clearPatterns() {
        singlePattern.clear();
        allPattern.clear();
        multiPattern.clear();
        allPattern_Index.clear();
        allPattern_reverseIndex.clear();
    }

    /**
     *
     * @param sentence -> List of SubSentWords objects of pushed-up or original subsentence
     * @param currentNode -> index of the current word being looked at
     * @param cand_set -> Set of indices of all patterns having prevNode word
     * @param wordBagTemp -> Map of current status of pattern words being crossed out
     * @return if atleast one pattern is matched or not
     */
    private static boolean treeSearch(List<SubSentWords> sentence, int currentNode, int prevNode, int entity,
                                      Set<Integer> cand_set, Map<Integer, List<String>> wordBagTemp){
        Set<Integer> candidateSet = new HashSet<>(cand_set);
        Map<Integer, List<String>> wordBag_temp = new HashMap<>(wordBagTemp);
        if (currentNode != entity) {
            if (!allPattern_reverseIndex.containsKey(sentence.get(currentNode).getLemma())) return false;
            candidateSet.retainAll(new HashSet<>(allPattern_reverseIndex.get(sentence.get(currentNode).getLemma())));
            if (candidateSet.isEmpty()) {
                return false;
            }
            for (Integer index : candidateSet) {
                if (wordBag_temp.containsKey(index)) {
                    List<String> temp = new ArrayList<>(wordBag_temp.get(index));
                    temp.remove(sentence.get(currentNode).getLemma());
                    wordBag_temp.put(index, temp);
                }
                if (wordBag_temp.get(index).size() == 0) {
                    return true;
                }
            }
        }
        boolean validPushUpScenario;
        for (int i = 65; i < 91; i++) {
            String ch = sentence.get(currentNode).getTrimmedEncoding() + (char)i;
            int child = returnEncodeIndex(sentence, ch);
            if (child != -1 && (prevNode == entity || child != prevNode)) {
                validPushUpScenario = treeSearch(sentence, child, currentNode, entity, candidateSet, wordBag_temp);
                if (validPushUpScenario) {
                    return true;
                }
            }
        }
        String currentWordEncoding = sentence.get(currentNode).getTrimmedEncoding();
        if (currentWordEncoding.length() < sentence.get(prevNode).getTrimmedEncoding().length()) {
            int parent = returnEncodeIndex(sentence, currentWordEncoding.substring(0, currentWordEncoding.length() - 1));
            if (parent != -1) {
                return treeSearch(sentence, parent, currentNode, entity, candidateSet, wordBag_temp);
            }
        }
        return false;
    }

    private static SubSentWords identify_entityLeaves(SubSentWords rootExp, SentenceProcessor wholeSentence){
        String root = rootExp.getLemma();
        List<SubSentWords> sentence = wholeSentence.getPushedUpSentences().getOrDefault(rootExp, wholeSentence.
                getSentenceBreakdown().get(rootExp));

        if (!allPattern_reverseIndex.containsKey(root)) return null;
        Set<Integer> root_index = new HashSet<>(allPattern_reverseIndex.get(root));
        List<String> typeSet = new ArrayList<>();
        List<Integer> specific_entities = new ArrayList<>();
        for (int i = 0; i < sentence.size(); i++){
            SubSentWords word = sentence.get(i);
            for (String ner : nerTypes){
                if (word.getLemma().contains(ner) && allPattern_reverseIndex.containsKey(word.getLemma())){
                    Set<Integer> entity_index = new HashSet<>(allPattern_reverseIndex.get(word.getLemma()));
                    entity_index.retainAll(root_index);
                    if (entity_index.size() > 0) {
                        typeSet.add(ner);
                        specific_entities.add(i);
                    }
                    // If number of entities in subsentence are greater than 1, skip the identification
                    if (typeSet.size() > 1) {
                        return null;
                    }
                }
            }
        }

        if (typeSet.isEmpty()) {
            return null;
        }

        List<Integer> entities = new ArrayList<>();
        int longest_encode = 0;
        for (int i : specific_entities){
            String subEnc = sentence.get(i).getTrimmedEncoding();
            if (subEnc.length() > longest_encode){
                entities.clear();
                entities.add(i);
                longest_encode = subEnc.length();
            } else if(subEnc.length() == longest_encode){
                entities.add(i);
            }
        }

        List<SubSentWords> merge_ent_encode = new ArrayList<>();
        for (Integer entity : entities){
            Set<Integer> candidate_set = new HashSet<>(allPattern_reverseIndex.get(sentence.get(entity).getLemma()));
            Map<Integer, List<String>> wordBag_temp = new HashMap<>(allPattern_Index);
            for (Integer index : candidate_set){
                if (wordBag_temp.containsKey(index)){
                    List<String> temp = new ArrayList<>(wordBag_temp.get(index));
                    temp.remove(sentence.get(entity).getLemma());
                    wordBag_temp.put(index,temp);
                }
            }
            boolean match_set = false;
            String currentWordEncoding = sentence.get(entity).getTrimmedEncoding();
            int parent = returnEncodeIndex(sentence, currentWordEncoding.substring(0, currentWordEncoding.length()-1));
            if (parent != -1) {
                match_set = treeSearch(sentence, parent, entity, entity, candidate_set, wordBag_temp);
            }
            if (match_set) {
                if (merge_ent_encode.size() == 1) {
                    return null;
                }
                merge_ent_encode.add(sentence.get(entity));
            }
        }
        return merge_ent_encode.size() == 1 ? merge_ent_encode.get(0) : null;
    }

    private static void hierarchicalPushUp(SentenceProcessor sentence) {
        sentence.resetPushUp();
        List<SubSentWords> sortedSubKeys = new ArrayList<>(sort_leafToTop(sentence));
        for (SubSentWords sub : sortedSubKeys) {
            boolean termReplaced = false;
            List<SubSentWords> subWords = new ArrayList<>(sentence.getSentenceBreakdown().get(sub));
            if (!sentence.getReplaceSurfaceName().isEmpty()) {
                for (int i = 0; i < subWords.size(); i++) {
                    String encode = subWords.get(i).getTrimmedEncoding();
                    if (sentence.getReplaceSurfaceName().containsKey(encode)) {
                        termReplaced = true;
                        SubSentWords replacement = new SubSentWords(sentence.getReplaceSurfaceName().get(encode));
                        replacement.setEncoding(subWords.get(i).getEncoding());
                        subWords.set(i, replacement);
                    }
                }
            }
            if (termReplaced) {
                sentence.pushUpSentences(sub, subWords);
            }
            SubSentWords entityLeaf = identify_entityLeaves(sub, sentence);
            if (entityLeaf != null) {
                sentence.updateReplacedSurfaceName(sub.getTrimmedEncoding(), entityLeaf);
            }
        }
    }

    private static void buildSentences() throws Exception {
        logger.log(Level.INFO, "STARTING: Sentences Processing");
        String inputFile = inputFolder + "annotated.txt";
        int totalNoOfSentences = totalLines(inputFile, noOfLines);
        logger.log(Level.INFO, "Total Number of Sentences: " + totalNoOfSentences);

        List<SentenceProcessor> sentenceCollector = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new FileReader(inputFile));
        String line = reader.readLine();

        int lineNo = 0;
        final boolean indexGiven = (line != null && line.split("\t").length != 1);
        int phraseCount = 0;
        Pattern pattern = Pattern.compile("[\\w\\d]+_[\\w\\d]+");
        while (line != null) {
            if (noOfLines > 0 && lineNo == noOfLines) {
                break;
            }
            if (!indexGiven || line.split("\t").length == 2) {
                String index = indexGiven ? line.split("\t")[0] : String.valueOf(lineNo);
                String sent = indexGiven ? line.split("\t")[1] : line;
                if (sent.split(" ").length > 4 && sent.split(" ").length < 100) {
                    if ( noOfBatches >= startBatch) {
                        Matcher matcher = pattern.matcher(sent);
                        Set<String> foundMatches = new HashSet<>();
                        while (matcher.find()) {
                            String match = matcher.group();
                            if (!foundMatches.contains(match)) {
                                foundMatches.add(match);
                                String newEntity = "PHRASEGEN" + phraseCount++;
                                sent = sent.replace(match, newEntity);
                                entityDictionary.put(newEntity, match);
                            }
                        }
                    }
                    sentenceCollector.add(new SentenceProcessor(sent, index));
                }
                if (sentenceCollector.size() == batchSize) {
                    if (noOfBatches >= startBatch) processParallelHelper(sentenceCollector, noOfBatches);
                    sentenceCollector.clear();
                    noOfBatches++;
                }
            }
            line = reader.readLine();
            lineNo++;
        }
        reader.close();

        if (sentenceCollector.size() != 0) {
            processParallelHelper(sentenceCollector, noOfBatches++);
            sentenceCollector.clear();
        }
        logger.log(Level.INFO, "COMPLETED: Sentences Processing");
    }

    private static void processParallelHelper(List<SentenceProcessor> sentenceCollector, int batchIterNo) throws Exception {
        IntStream istream = IntStream.range(0, sentenceCollector.size());
        forkJoinPool.submit(() -> istream.parallel().forEach(s -> sentenceCollector.get(s).processSentence(pipeline, entityDictionary, nerTypes))
        ).get();
        istream.close();
        logger.log(Level.INFO, "PROCESSED: Batch Iteration Number - " + batchIterNo);
        saveSentenceBreakdown(batchIterNo, sentenceCollector, -1);
    }

    private static void buildDictionary() throws Exception {
        logger.log(Level.INFO, "STARTING: Dictionary Building");
        ObjectMapper mapper = new ObjectMapper();
        entityDictionary = mapper.readValue(new File(
                    inputFolder + "dict.json"), new TypeReference<Map<String, String>>() {});
        logger.log(Level.INFO, "COMPLETED: Dictionary Building");
    }

    private static void saveSentenceBreakdown(int batchIterNo, List<SentenceProcessor> sentenceCollector, int tempIter) throws Exception {
        logger.log(Level.INFO, "STARTING: Sentence Breakdown Serialization for batch " + batchIterNo);
        String directory = inputFolder + "ProcessedInput/" + batchSize + "/";
        String prefix = tempIter == -1 ? "sentenceBatch"  + batchIterNo + "." + noOfLines : "t" + batchIterNo;
        File file = new File(directory + prefix + (tempIter == -1 ? "" : tempIter) + ".txt");
        FileOutputStream fileOut = new FileOutputStream(file);
        ObjectOutputStream out = new ObjectOutputStream(fileOut);
        if (tempIter != -1) file.deleteOnExit();
        if (tempIter > 2) {
            File file1 = new File(directory + prefix  + (tempIter - 1) + ".txt");
            if (file1.delete()) logger.log(Level.INFO, "Deleted previous iteration file of batch " + batchIterNo);
            else logger.log(Level.INFO, "Could not delete previous iteration file of batch " + batchIterNo);
        }
        out.writeObject(sentenceCollector);
        out.close();
        fileOut.close();
        logger.log(Level.INFO, "COMPLETED: Sentence Breakdown Serialization for batch " + batchIterNo);
    }

    private static List<SentenceProcessor> loadSentenceBreakdown(int batchIterNo, int tempIter) throws Exception {
        logger.log(Level.INFO, "STARTING: Sentence Breakdown Deserialization for batch " + batchIterNo);
        String directory = inputFolder + "ProcessedInput/" + batchSize + "/";
        String prefix = tempIter == -1 ? "sentenceBatch" + batchIterNo + "." + noOfLines : "t" + batchIterNo + tempIter;
        FileInputStream fileIn = new FileInputStream( directory + prefix + ".txt");
        ObjectInputStream in = new ObjectInputStream(fileIn);
        List<SentenceProcessor> sentenceCollector = (List<SentenceProcessor>) in.readObject();
        in.close();
        fileIn.close();
        logger.log(Level.INFO, "COMPLETED: Sentence Breakdown Deserialization for batch " + batchIterNo);
        return sentenceCollector;
    }

    private static void savePatternClassificationData() throws IOException{
        logger.log(Level.INFO, "STARTING: Saving Meta-Pattern Classification Data");
        String suffix = "." + noOfLines + "_" + minimumSupport + ".txt";
        FileWriter singlePatterns = new FileWriter(outputFolder + "singlePatterns" + suffix);
        FileWriter multiPatterns = new FileWriter(outputFolder + "multiPatterns" + suffix);
        FileWriter allPatterns = new FileWriter(outputFolder + "allPatterns" + suffix);
        writePatternsToFile(new BufferedWriter(singlePatterns), singlePattern);
        writePatternsToFile(new BufferedWriter(multiPatterns), multiPattern);
        writePatternsToFile(new BufferedWriter(allPatterns), allPattern);
        logger.log(Level.INFO, "COMPLETED: Saving Meta-Pattern Classification Data");
    }

    private static void loadPatternClassificationData() throws IOException {
        logger.log(Level.INFO, "STARTING: Meta-Patterns Loading and Classification");
        String name = "allPatterns";
        String directory = outputFolder;
        String suffix = "." + noOfLines + "_" + minimumSupport + ".txt";
        BufferedReader patterns = new BufferedReader(new FileReader(directory + name + suffix));
        String line = patterns.readLine();
        frequentPatterns = new ArrayList<>();
        while (line != null) {
            String[] data = line.split("->");
            frequentPatterns.add(new MetaPattern(data[0], nerTypes, stopWords, Integer.parseInt(data[1])));
            line = patterns.readLine();
        }
        pattern_classification();
        logger.log(Level.INFO, "COMPLETED: Meta-Patterns Loading and Classification");
    }

    private static String patternFinding(SentenceProcessor sentence, List<Map<String, Integer>> patternList) throws Exception {
        try {
            List<PatternInstance> ans = new ArrayList<>();
            List<PatternInstance> output;
            Map<SubSentWords, List<SubSentWords>> map;
            for (SubSentWords subRoot : sentence.getSentenceBreakdown().keySet()) {
                int iter = 1;
                if (sentence.getPushedUpSentences().containsKey(subRoot)) {
                    iter++;
                }
                while (iter > 0) {
                    map = (iter == 2) ? sentence.getPushedUpSentences() : sentence.getSentenceBreakdown();
                    output = patternMatchingHelper(sentence, map, subRoot, patternList);
                    if (output != null) ans.addAll(output);
                    iter--;
                }
            }
            String answer = null;
            if (!ans.isEmpty()) {
                ans = ans.stream().sorted(Comparator.comparing(e -> e.getAllElementIndices().size())).distinct().collect(Collectors.toList());
                List<PatternInstance> finalAns = new ArrayList<>();
                boolean redundant;
                for (int i = 0; i < ans.size() - 1; i++) {
                    redundant = false;
                    for (int j = i + 1; j < ans.size(); j++) {
                        if (ans.get(j).getAllElementIndices().containsAll(ans.get(i).getAllElementIndices())) {
                            redundant = true;
                            break;
                        }
                    }
                    if (!redundant) finalAns.add(ans.get(i));
                }
                finalAns.add(ans.get(ans.size() - 1));
                answer = includeContext ? createExtractionString(finalAns) : finalAns.stream().map(PatternInstance::toString).collect(Collectors.joining(""));
            }
            return answer;
        } catch (Exception e) {
            logger.log(Level.WARNING,"ERROR in " + sentence.getSentID());
            throw new Exception(e);
        }
    }

    private static List<PatternInstance> patternMatchingHelper(SentenceProcessor sentence, Map<SubSentWords, List<SubSentWords>> dict,
                                                SubSentWords subRoot, List<Map<String, Integer>> patternList) {
        List<PatternInstance> out = new ArrayList<>();
        int multiCount = 0;
        List<SubSentWords> subSent = dict.get(subRoot);
        int entityCount = noOfEntities(subSent, nerTypes);
        for (int i = 0; i < patternList.size(); i++) {
            if (entityCount == 0 || (i == 0 && entityCount < 2)) continue;
            int endIndex = -1, startIndex = subSent.size();
            for (String metaPattern : patternList.get(i).keySet()) {
                if (i != 0 && (multiCount > 0 || metaPattern.replaceAll("[_\\-]+", " ").split(" ").length < 3)) break;
                PatternMatchIndices matchFound = check_subsequence(subSent, true, metaPattern, nerTypes);
                if (matchFound != null) {
                    int newStart = matchFound.getEntityIndices().get(0), newEnd = matchFound.getEntityIndices().get(matchFound.getEntityIndices().size() - 1);
                    boolean check1 = (i != 0) ? (newStart > endIndex) : (newStart >= endIndex);
                    boolean check2 = (i != 0) ? (newEnd < startIndex) : (newEnd <= startIndex);
                    if (check1 || check2) {
                        if (i == 0) multiCount++;
                        startIndex = Math.min(startIndex, newStart);
                        endIndex = Math.max(endIndex, newEnd);
                        PatternInstance instance = new PatternInstance(sentence, subRoot, metaPattern,
                                matchFound, nerTypes);
                        out.add(instance);
                        out.addAll(instance.generateAlternatePattern());
                    }
                }
            }
        }
        return out.isEmpty() ? null : out;
    }

    private static void savePatternMatchingResults(List<Map<String, Integer>> patternList) throws Exception{
        logger.log(Level.INFO, "STARTING: Pattern Matching");
        String outputDirectory = outputFolder;
        String suffix = "." + noOfLines + "_" + minimumSupport + ".txt";
        BufferedWriter patternOutput = new BufferedWriter(new FileWriter(outputDirectory + "c_patternOutput" + suffix));
        for (int batchNo = startBatch; batchNo < noOfBatches; batchNo++) {
            List<SentenceProcessor> sentenceCollectorBatch = loadSentenceBreakdown(batchNo, noOfPushUps + 1);
            String[] outputArray = new String[sentenceCollectorBatch.size()];
            forkJoinPool.submit(() -> IntStream.range(0, sentenceCollectorBatch.size()).parallel()
                    .forEach(s -> {
                        try {
                            outputArray[s] = patternFinding(sentenceCollectorBatch.get(s), patternList);
                        } catch (Exception e) {
                            StringWriter errors = new StringWriter();
                            e.printStackTrace(new PrintWriter(errors));
                            logger.log(Level.WARNING, errors.toString());
                        }
                    })
            ).get();
            Arrays.stream(outputArray).forEachOrdered(s -> {
                try {
                    if (s != null) patternOutput.write(s);
                } catch (IOException e) {
                    StringWriter errors = new StringWriter();
                    e.printStackTrace(new PrintWriter(errors));
                    logger.log(Level.SEVERE, errors.toString());
                }
            });
            sentenceCollectorBatch.clear();
            logger.log(Level.INFO, "PROCESSED: Batch Iteration Number - " + batchNo);
        }
        patternOutput.close();
        logger.log(Level.INFO, "COMPLETED: Pattern Matching");
    }

    public static void call() {
        try {
            initialization();
            if (!load_sentenceBreakdownData) {
//                buildDictionary();
                buildSentences();
            }
            int prevPatternCount = 0;
            int iterations = 1;

            List<Map<String, Integer>> patternList = new ArrayList<>();
            if (load_metapatternData) {
                loadPatternClassificationData();
                noOfPushUps = 1;
            } else {
                logger.log(Level.INFO, "STARTING: Iterative Frequent Pattern Mining followed by Hierarchical Pushups");
                frequent_pattern_mining(iterations);
            }

            while (iterations <= noOfPushUps && prevPatternCount < allPattern.size()) {
                prevPatternCount = allPattern.size();
                logger.log(Level.INFO, "STARTING: Hierarchical PushUps - Iteration " + iterations);
                for (int batchNo = startBatch; batchNo < noOfBatches; batchNo++) {
                    List<SentenceProcessor> sentenceCollectorBatch = loadSentenceBreakdown(batchNo, iterations == 1 ? -1 : iterations);
                    forkJoinPool.submit(() -> sentenceCollectorBatch.parallelStream().forEach(CPWW::hierarchicalPushUp)).get();
                    saveSentenceBreakdown(batchNo, sentenceCollectorBatch, iterations + 1);
                    sentenceCollectorBatch.clear();
                    logger.log(Level.INFO, "PROCESSED: Batch Iteration Number - " + batchNo);
                }
                logger.log(Level.INFO, "COMPLETED: Hierarchical PushUps - Iteration " + iterations++);
                if (!load_metapatternData) frequent_pattern_mining(iterations);
            }
            if (iterations <= noOfPushUps) noOfPushUps = iterations - 1;
            savePatternClassificationData();
            patternList.add(returnSortedPatternList(multiPattern));
            if (includeContext) patternList.add(returnSortedPatternList(singlePattern));
            savePatternMatchingResults(patternList);
        } catch (Exception e) {
            StringWriter errors = new StringWriter();
            e.printStackTrace(new PrintWriter(errors));
            logger.log(Level.SEVERE, errors.toString());
        } finally {
            if (forkJoinPool != null) {
                forkJoinPool.shutdown();
            }
        }
    }
}