package cpww;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;

import java.io.*;
import java.util.*;
import java.util.logging.FileHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;
import java.util.stream.Collectors;

import static cpww.Util.*;

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

    private static String[] nerTypes;
    private static List<String> stopWords;
    private static List<SentenceProcessor> sentenceCollector;
    private static Map<String, String> entityDictionary = new HashMap<>();
    private static List<MetaPattern> frequentPatterns = new ArrayList<>();
    private static Map<String, List<MetaPattern>> singlePattern = new HashMap<>();
    private static Map<String, List<MetaPattern>> allPattern = new HashMap<>();
    private static Map<String, List<MetaPattern>> multiPattern = new HashMap<>();
    private static Map<Integer, List<String>> allPattern_Index = new HashMap<>();
    private static Map<Integer,List<String>> wordBag_temp = new HashMap<>();
    private static Map<String, Set<Integer>> allPattern_reverseIndex = new HashMap<>();

    private static Logger logger = Logger.getLogger(CPWW.class.getName()) ;


    public CPWW()  {
    }

    private static void initialization() throws IOException {
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
        inputFolder = prop.getProperty("inputFolder");
        outputFolder = prop.getProperty("outputFolder");
        noOfLines = Integer.parseInt(prop.getProperty("noOfLines"));
        minimumSupport = Integer.parseInt(prop.getProperty("minimumSupport"));
        maxLength = Integer.parseInt(prop.getProperty("maxLength"));
        nerTypes = readList(new FileReader(prop.getProperty("nerTypes"))).toArray(new String[0]);
        stopWords = readList(new FileReader(prop.getProperty("stopWordsFile")));
        load_sentenceBreakdownData = Boolean.parseBoolean(prop.getProperty("load_sentenceBreakdownData"));
        load_metapatternData = Boolean.parseBoolean(prop.getProperty("load_metapatternData"));

        sentenceCollector = new ArrayList<>();
        FileHandler logFile = new FileHandler(inputFolder + data_name + "_logFile.txt");
        logFile.setFormatter(new SimpleFormatter());
        logger.addHandler(logFile);
        File dir = new File(inputFolder);
        if (!dir.exists()) {
            throw new IOException("Input Folder not found.");
        }
        dir = new File(outputFolder);
        if (!dir.exists()) {
            dir.mkdir();
        }
    }

//    public String replacedNER(String sentence) {
//        String[] arr = sentence.replaceAll("\n","").split(" ");
//        if (arr.length <5 || !Character.isLetter(sentence.charAt(0))) {
//            return null;
//        }
//        for (int i = 0; i < arr.length; i++) {
//            String orig = arr[i].replaceAll("([A-Z]+_[A-Z0-9)]+_[\\w']+(._)*[\\w']*)\\s*", "$1");
//            Pattern pattern = Pattern.compile("_[\\w']+(._)*[\\w']*");
//            Matcher matcher = pattern.matcher(orig);
//            arr[i] = orig.replaceAll("([A-Z]+)_[A-Z0-9)]+_[\\w']+(._)*[\\w']*","$1");
//
//            for (String ner : nerTypes) {
//                if (arr[i].contains(ner.toUpperCase())) {
//                    arr[i] = arr[i].replaceAll(ner.toUpperCase(), ner.toUpperCase() + (++nerCount));
//                    if (matcher.find()) {
//                        String match = matcher.group().replaceFirst("_[A-Z0-9]+_","");
//                        this.entityDictionary.put(ner.toUpperCase() + nerCount, match);
//                    }
//                    break;
//                }
//            }
//        }
//        return String.join(" ", arr).replaceAll("(_)+", "-");
//    }

    private static void frequent_pattern_mining(int iteration) throws  IOException{
        frequentPatterns.clear();

        List<SubSentWords> tokens = new ArrayList<>();
        Map<String, List<Integer>> dict_token = new HashMap<>();
        Map<String, List<Integer>> valid_pattern = new HashMap<>();
        int tokenNumber = 0;
        String token;
        for (SentenceProcessor sentence : sentenceCollector) {
            Map<SubSentWords, List<SubSentWords>> sentBreak = sentence.getSentenceBreakdown();
            for (SubSentWords key : sentBreak.keySet()) {
                List<SubSentWords> list = sentBreak.get(key);
                for (SubSentWords word : list) {
                    token = word.getLemma();
                    tokens.add(word);
                    updateMapCount(dict_token, token, tokenNumber++);
                    updateMapCount(valid_pattern, token, 1);
                }
                tokens.add(new SubSentWords("$","$", "", -1));
                tokenNumber++;
            }
            for (SubSentWords key : sentence.getPushedUpSentences().keySet()) {
                for (SubSentWords word : sentence.getPushedUpSentences().get(key)) {
                    token = word.getLemma();
                    tokens.add(word);
                    updateMapCount(dict_token, token, tokenNumber++);
                    updateMapCount(valid_pattern, token, 1);
                }
                tokens.add(new SubSentWords("$","$", "", -1));
                tokenNumber++;
            }
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

    private static void pattern_classification() throws IOException {
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
     * @param sentence -> Subsentence being looked at
     * @param currentNode -> SubSentWords object of the current word being looked at
     * @param prevNode -> SubSentWords object of the previous word that was being looked at
     * @param cand_set -> Set of indices of all patterns having prevNode word
     * @param match_set -> Set of indices of all complete patterns found
     * @return set of indices of all complete patterns found
     */
    private static Set<Integer> treeSearch(List<SubSentWords> sentence, SubSentWords currentNode, SubSentWords prevNode,
                                           Set<Integer> cand_set, Set<Integer> match_set){
        Set<Integer> matching_set = new HashSet<>(match_set);
        if (!allPattern_reverseIndex.containsKey(currentNode.getLemma())) {
            return matching_set;
        }
        Set<Integer> candidateSet = new HashSet<>(cand_set);
        candidateSet.retainAll(new HashSet<>(allPattern_reverseIndex.get(currentNode.getLemma())));
        if (candidateSet.size() == 0) {
            return matching_set;
        }
        for (Integer index : candidateSet){
            if (wordBag_temp.containsKey(index)){
                List<String> temp = new ArrayList<>(wordBag_temp.get(index));
                temp.remove(currentNode.getLemma());
                wordBag_temp.put(index,temp);
            }
            if (wordBag_temp.get(index).size() == 0) {
                matching_set.add(index);
            }
        }
        candidateSet.removeAll(matching_set);

        for (int i = 65; i < 91; i++){
            String ch = currentNode.getTrimmedEncoding() + (char)i;
            SubSentWords child = returnSubSentWord(sentence, ch);
            if (child == null)    break;
            if (!child.getTrimmedEncoding().equals(prevNode.getTrimmedEncoding())) {
                matching_set = new HashSet<>(treeSearch(sentence, child, currentNode, candidateSet, matching_set));
            }
        }
        if (currentNode.getTrimmedEncoding().length() < prevNode.getTrimmedEncoding().length()){
            String parent = currentNode.getTrimmedEncoding().substring(0, currentNode.getTrimmedEncoding().length()-1);
            SubSentWords temp = returnSubSentWord(sentence, parent);
            if (temp != null) {
                matching_set = new HashSet<>(treeSearch(sentence, temp, currentNode, candidateSet, matching_set));
            }
        }
        return matching_set;
    }

    private static SubSentWords identify_entityLeaves(SubSentWords rootExp, List<SubSentWords> sentence){
        String root = rootExp.getLemma();

        if (!allPattern_reverseIndex.containsKey(root)) return null;
        Set<Integer> root_index = new HashSet<>(allPattern_reverseIndex.get(root));
        Set<String> typeSet = new HashSet<>();
        List<Integer> specific_entities = new ArrayList<>();
        for (int i = 0; i < sentence.size(); i++){
            String word = sentence.get(i).getLemma();
            for (String ner : nerTypes){
                if (word.contains(ner) && allPattern_reverseIndex.containsKey(word)){
                    Set<Integer> entity_index = new HashSet<>(allPattern_reverseIndex.get(word));
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

        List<SubSentWords> entities = new ArrayList<>();
        int longest_encode = 0;
        for (int i : specific_entities){
            SubSentWords subEnc = sentence.get(i);
            if (subEnc.getTrimmedEncoding().length() > longest_encode){
                entities.clear();
                entities.add(subEnc);
            } else if(subEnc.getTrimmedEncoding().length() == longest_encode){
                entities.add(subEnc);
            }
        }

        Set<Integer> merge_match_set = new HashSet<>();
        List<SubSentWords> merge_ent_encode = new ArrayList<>();
        for (SubSentWords entity : entities){
            Set<Integer> candidate_set = new HashSet<>(allPattern_reverseIndex.get(entity.getLemma()));
            wordBag_temp = new HashMap<>(allPattern_Index);
            for (Integer index : candidate_set){
                if (wordBag_temp.containsKey(index)){
                    List<String> temp = new ArrayList<>(wordBag_temp.get(index));
                    temp.remove(entity.getLemma());
                    wordBag_temp.put(index,temp);
                }
            }
            Set<Integer> match_set = new HashSet<>();
            String parentEnc = entity.getTrimmedEncoding().substring(0, entity.getTrimmedEncoding().length()-1);
            for (SubSentWords word : sentence) {
                if (word.getTrimmedEncoding().equals(parentEnc)) {
                    match_set = treeSearch(sentence, word, entity, candidate_set, match_set);
                    break;
                }
            }

            merge_match_set.addAll(match_set);
            if (match_set.size() > 0) {
                merge_ent_encode.add(entity);
            }
        }
        if (merge_match_set.size() != 1) {
            return null;
        }
        return merge_ent_encode.get(0);
    }

    private static List<String> hierarchical_expansion(SentenceProcessor sentence, String original_subEnc,
                                                       SubSentWords entityEncode){
        List<SubSentWords> added_subEnc = sentence.getSentenceBreakdown().get(entityEncode);
        SubSentWords replace_encode = sentence.getReplaceSurfaceName().get(entityEncode.getTrimmedEncoding());
        String entity_encode = entityEncode.getEncoding();
        int index_diff = original_subEnc.length() - entity_encode.length();

        List<String> ans = new ArrayList<>();
        ans.add(replace_encode.getEncoding());
        if (original_subEnc.contains(replace_encode.getEncoding()) || replace_encode.equals(entityEncode)) {
            // Do nothing
        } else if (original_subEnc.contains(" " + entity_encode + " ")) {
            original_subEnc = original_subEnc.replace(" " + entity_encode + " ", " {{" + added_subEnc + "}} ");
        } else if (original_subEnc.contains("{{" + entity_encode + " ")) {
            original_subEnc = original_subEnc.replace("{{" + entity_encode + " ", "{{{{" + added_subEnc + "}} ");
        } else if (original_subEnc.contains(" " + entity_encode + "}}")) {
            original_subEnc = original_subEnc.replace(" " + entity_encode + "}}", " {{" + added_subEnc + "}}}}");
        } else if (original_subEnc.contains("{{" + entity_encode + "}}")) {
            original_subEnc = original_subEnc.replace("{{" + entity_encode + "}}", "{{{{" + added_subEnc + "}}}}");
        } else if (index_diff > 0 && original_subEnc.substring(index_diff - 1).contains(" " + entity_encode)) {
            original_subEnc = original_subEnc.substring(0, index_diff - 1) + " {{" + added_subEnc + "}}";
        } else if (original_subEnc.contains(entity_encode + " ")) {
            original_subEnc = "{{" + added_subEnc + "}} " + original_subEnc.substring(entity_encode.length() + 1);
        }
        if (!sentence.getReplaceSurfaceName().containsKey(replace_encode.getTrimmedEncoding()) ||
                replace_encode.equals(entityEncode)) {
            ans.add(original_subEnc);
            return ans;
        }
        return hierarchical_expansion(sentence, original_subEnc, replace_encode);
    }

    public static String map_original_tokens(SentenceProcessor sentence, String encode){
        Map<String, SubSentWords> encodingMap = sentence.getReverseWordEncoding();
        String[] enc = encode.split(" ");
        List<String> ans = new ArrayList<>();
        for (String encoding : enc){
            String temp = "", res = ""; boolean flag = false;
            for (int i = 0; i < encoding.length(); i++){
                if (res.equals("") && encoding.charAt(i) == '{'){
                    temp += "{";
                }
                else if(!res.equals("") && encoding.charAt(i) == '}' && !flag){
                    res = temp + encodingMap.get(res).getOriginalWord() + "}";
                    flag = true;
                }
                else if (!res.equals("") && encoding.charAt(i) == '}' && flag) res += "}";
                else res += encoding.charAt(i);
            }
            if (!flag) res = temp + encodingMap.get(res).getOriginalWord();
            ans.add(res);
        }
        return String.join(" ", ans);
    }

    private static void buildSentences() throws Exception {
        logger.log(Level.INFO, "STARTING: Sentences Processing");
        String inputFile = inputFolder + data_name + "_annotated.txt";
        BufferedReader reader = new BufferedReader(new FileReader(inputFile));
        String line = reader.readLine();
        boolean indexGiven = false;
        int lineNo = 1;
        if (line != null && line.split("\t").length != 1) {
            indexGiven = true;
        }
        while (line != null) {
            if (noOfLines > 0 && lineNo == noOfLines + 1) {
                break;
            }
            String sentence = indexGiven ? line.split("\t")[1] : line;
            String index = indexGiven ? line.split("\t")[0] : String.valueOf(lineNo);

            if (sentence != null && sentence.split(" ").length < 100) {
                SentenceProcessor sp = new SentenceProcessor(pipeline, nerTypes, entityDictionary, sentence, index);
                sentenceCollector.add(sp);
            }
            line = reader.readLine();
            lineNo++;
        }
        reader.close();
        logger.log(Level.INFO, "COMPLETED: Sentences Processing");
    }

    private static void buildDictionary() throws Exception {
        logger.log(Level.INFO, "STARTING: Dictionary Building");
        ObjectMapper mapper = new ObjectMapper();
        entityDictionary = mapper.readValue(new File(
                    inputFolder + data_name + "_dict.json"), new TypeReference<Map<String, String>>() {});
        logger.log(Level.INFO, "COMPLETED: Dictionary Building");
    }

    private static void saveSentenceBreakdown() throws Exception {
        logger.log(Level.INFO, "STARTING: Sentence Breakdown Serialization");
        String directory = inputFolder + data_name;
        FileOutputStream fileOut = new FileOutputStream(directory + "_sentenceBreakdown." + noOfLines + ".txt");
        ObjectOutputStream out = new ObjectOutputStream(fileOut);
        out.writeObject(sentenceCollector);
        fileOut.close();
        logger.log(Level.INFO, "COMPLETED: Sentence Breakdown Serialization");
    }

    private static void loadSentenceBreakdown() throws Exception {
        logger.log(Level.INFO, "STARTING: Sentence Breakdown Deserialization");
        String directory = inputFolder + data_name;
        FileInputStream fileIn = new FileInputStream(directory + "_sentenceBreakdown." + noOfLines + ".txt");
        ObjectInputStream in = new ObjectInputStream(fileIn);
        sentenceCollector = (List<SentenceProcessor>) in.readObject();
        fileIn.close();
        logger.log(Level.INFO, "COMPLETED: Sentence Breakdown Deserialization");
    }

    private static void savePatternClassificationData() throws IOException{
        logger.log(Level.INFO, "STARTING: Saving Meta-Pattern Classification Data");
        String directory = outputFolder + data_name;
        String suffix = "." + noOfLines + "_" + minimumSupport + ".txt";
        FileWriter singlePatterns = new FileWriter(directory + "_singlePatterns" + suffix);
        FileWriter multiPatterns = new FileWriter(directory + "_multiPatterns" + suffix);
        FileWriter allPatterns = new FileWriter(directory + "_allPatterns" + suffix);
        writePatternsToFile(new BufferedWriter(singlePatterns), singlePattern);
        writePatternsToFile(new BufferedWriter(multiPatterns), multiPattern);
        writePatternsToFile(new BufferedWriter(allPatterns), allPattern);
        logger.log(Level.INFO, "COMPLETED: Saving Meta-Pattern Classification Data");
    }

    private static SentenceProcessor hierarchicalPushUp(SentenceProcessor sentence) {
        sentence.resetPushUp();
        for (Map.Entry<SubSentWords, List<SubSentWords>> sub : sort_leafToTop(sentence)) {
            boolean termReplaced = false;
            List<SubSentWords> subWords = sub.getValue();
            if (!sentence.getReplaceSurfaceName().isEmpty()) {
                for (int i = 0; i < subWords.size(); i++) {
                    String encode = subWords.get(i).getTrimmedEncoding();
                    if (sentence.getReplaceSurfaceName().containsKey(encode)) {
                        termReplaced = true;
                        subWords.set(i, sentence.getReplaceSurfaceName().get(encode));
                    }
                }
            }
            if (termReplaced) {
                sentence.pushUpSentences(sub.getKey(), subWords);
            }
            SubSentWords entityLeaf = identify_entityLeaves(sub.getKey(), subWords);
            sub.setValue(subWords);
            if (entityLeaf != null) {
                sentence.updateReplacedSurfaceName(sub.getKey().getTrimmedEncoding(), entityLeaf);
            }
        }
    return sentence;
    }

    private static List<String> patternFinding(SentenceProcessor sentence) {
        List<String> ans = new ArrayList<>();
        String output;
        Map<SubSentWords, List<SubSentWords>> map;
        Map<String, List<MetaPattern>> freqPatterns;
        for (SubSentWords subRoot : sentence.getSentenceBreakdown().keySet()) {
            int iter = 2;
            if (sentence.getPushedUpSentences().containsKey(subRoot)) {
                iter += 2;
            }
            while (iter > 0) {
                map = (iter > 2) ? sentence.getPushedUpSentences() : sentence.getSentenceBreakdown();
                freqPatterns = (iter % 2 == 0) ? singlePattern : multiPattern;
                output = patternMatchingHelper(sentence, map, subRoot, freqPatterns);
                if (output != null) ans.add(output);
                iter--;
            }
        }
        return ans;
    }

    private static String patternMatchingHelper(SentenceProcessor sentence, Map<SubSentWords, List<SubSentWords>> dict,
                                                SubSentWords subRoot, Map<String, List<MetaPattern>> patternList) {
        int noOfEntityTypes = 0;
        int longestMatch = 0;
        String matchedPattern = "";
        List<Integer> matchedEntityPos = new ArrayList<>();
        for (String metaPattern : patternList.keySet()) {
            List<MetaPattern> metaPatternsList = patternList.get(metaPattern);
            if (maxNerCount(metaPatternsList) > noOfEntityTypes || (maxNerCount(metaPatternsList) == noOfEntityTypes &&
                    metaPattern.length() > longestMatch)) {
                if (dict.containsKey(subRoot)) {
                    List<Integer> nerIndices = check_subsequence(dict.get(subRoot), metaPattern, nerTypes);
                    if (nerIndices != null) {
                        longestMatch = metaPattern.length();
                        noOfEntityTypes = maxNerCount(metaPatternsList);
                        matchedPattern = metaPattern;
                        matchedEntityPos = new ArrayList<>(nerIndices);
                    }
                }
            }
        }
        return generatePatternInstances(sentence, subRoot, matchedPattern, matchedEntityPos);
    }

    private static String generatePatternInstances(SentenceProcessor sentence, SubSentWords subRoot,
                                                   String matchedPattern, List<Integer> entityPos) {
        if (matchedPattern.equals("")) return null;
        List<String> instances = new ArrayList<>();
        Map<String, SubSentWords> encodingMap = sentence.getReverseWordEncoding();
        int adjust_length = 0;
        String words = null;
        List<SubSentWords> value = sentence.getSentenceBreakdown().get(subRoot);
        String subEnc = value.stream().map(SubSentWords::getEncoding).collect(Collectors.joining(" "));
        for (Integer entityPo : entityPos) {
            SubSentWords entity_code = value.get(entityPo + adjust_length);
            if (sentence.getReplaceSurfaceName().containsKey(entity_code.getTrimmedEncoding()) &&
                    !entity_code.getTrimmedEncoding().equals(subRoot.getTrimmedEncoding())) {
                List<String> output = hierarchical_expansion(sentence, subEnc, entity_code);
                adjust_length += output.get(1).split(" ").length - subEnc.split(" ").length;
                subEnc = output.get(1);
                words = map_original_tokens(sentence, subEnc);
                instances.add(output.get(0));
            } else {
                words = map_original_tokens(sentence, subEnc);
                instances.add(entity_code.getEncoding());
            }
        }
        if (words != null) {
            String output = sentence.getSentID() + "\t" + matchedPattern + "\t[";
            for (String instance : instances) {
                output += encodingMap.get(instance).getOriginalWord() + ", ";
            }
            output = output.substring(0, output.length() - 2) + "]\t" + words + "\n";
            return output;
        }
        return null;
    }

    public static void call() throws Exception{
        initialization();
        String inputDirectory = inputFolder + data_name;
        String outputDirectory = outputFolder + data_name;

        boolean metadata_exists = new File(inputDirectory + "_dict.json").exists();
        boolean annotatedDataExists = new File(inputDirectory + "_annotated.txt").exists();
        if (!metadata_exists || !annotatedDataExists) {
            throw new Exception("Required Files not found.");
        }
        String suffix = "." + noOfLines + ".txt";
        boolean breakdownDataExists = new File(inputDirectory + "_sentenceBreakdown" + suffix).exists();
        if (load_sentenceBreakdownData && breakdownDataExists) {
            loadSentenceBreakdown();
        } else if (load_sentenceBreakdownData && !breakdownDataExists) {
            throw new IOException("Sentence Breakdown Data does not exist.");
        } else {
            buildDictionary();
            buildSentences();
            saveSentenceBreakdown();
        }
        int prevPatternCount = 0;
        int iterations = 1;

        logger.log(Level.INFO, "STARTING: Iterative Frequent Pattern Mining and Hierarchical Pushups");
        frequent_pattern_mining(iterations);
        while (prevPatternCount < allPattern.size()) {
            prevPatternCount = allPattern.size();
            logger.log(Level.INFO, "STARTING: Hierarchical PushUps - Iteration " + iterations);
            for (int i = 0; i < sentenceCollector.size(); i++) {
                sentenceCollector.set(i, hierarchicalPushUp(sentenceCollector.get(i)));
            }
            logger.log(Level.INFO, "COMPLETED: Hierarchical PushUps - Iteration " + iterations);
            frequent_pattern_mining(++iterations);
        }

        savePatternClassificationData();

        logger.log(Level.INFO, "STARTING: Pattern Matching");

        suffix = "." + noOfLines + "_" + minimumSupport + ".txt";
        BufferedWriter patternOutput = new BufferedWriter(new FileWriter(outputDirectory + "_patternOutput" + suffix));
        for (SentenceProcessor sp : sentenceCollector) {
            for (String output : patternFinding(sp)) {
                patternOutput.write(output);
            }
        }
        patternOutput.close();
        logger.log(Level.INFO, "COMPLETED: Pattern Matching");
    }
}