package cpww.utils;

import cpww.MetaPattern;
import cpww.PatternInstance;
import cpww.SentenceProcessor;
import cpww.SubSentWords;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.process.Morphology;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;

import java.io.*;
import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class Util {
    public static final String[] articles = new String[]{"a", "an", "the"};
    public static final String patternRegexFilter = "[`$\\d%'\\.,\\*/\\(\\):;\\!\\-\\|\"\\\\]+(lrb|lsb|rrb|rsb)*[`$\\d%'\\.,\\*/\\(\\):;\\!\\-\\|\"\\\\]*";
    public static final Pattern pattern = Pattern.compile("[A-Z]+[\\d]+");


    public static int sum(List<Integer> abc){
        int total = 0;
        for (Integer integer : abc) total += integer;
        return total;
    }

    public static String min_characters(List<String> abc){
        String ans = abc.get(0);
        int total = ans.length();
        for (int i = 1; i < abc.size(); i++){
            if (abc.get(i).length() < total) {
                total = abc.get(i).length();
                ans = abc.get(i);
            }
        }
        return ans;
    }

    public static List<Map.Entry<String, Integer>> sortDecreasing(Map<String, Integer> map) {
        List<Map.Entry<String, Integer>> l = new ArrayList<>(map.entrySet());
        // Arrange patterns in decreasing order of frequency
        l.sort((o1, o2) -> o2.getValue().compareTo(o1.getValue()));
        return l;
    }

    public static PatternMatchIndices check_subsequence(List<SubSentWords> main_sequence, boolean checkContinuity,
                                           String mainPattern, String[] nerTypes){
        String[] pattern = mainPattern.split(" ");
        int m = main_sequence.size(), n = pattern.length;
        List<Integer> entityIndices = new ArrayList<>();
        List<Integer> actualSentLevelElementIndices = new ArrayList<>();
        List<Integer> storingIndex = new ArrayList<>();
        List<String> patternTree = new ArrayList<>();
        boolean solutionFound;
        int startingIndex = 0;
        while (true) {
            int i = 0;
            solutionFound = true;
            for (int j = startingIndex ; j < m && i < n; j++) {
                if (main_sequence.get(j).getLemma().equals(pattern[i])) {
                    storingIndex.add(j);
                    actualSentLevelElementIndices.add(main_sequence.get(j).getIndex());
                    patternTree.add(main_sequence.get(j).getTrimmedEncoding());
                    if (containsEntity(main_sequence.get(j).getLemma(), nerTypes)) {
                        entityIndices.add(j);
                    }
                    i++;
                }
            }
            if (i == n) {
                if (checkContinuity) {
                    String root = min_characters(patternTree);
                    for (String t : patternTree) {
                        String parent = t.substring(0, t.length() - 1);
                        if (!patternTree.contains(parent) && !t.equals(root)) {
                            solutionFound = false;
                            break;
                        }
                    }
                    if (solutionFound && noConjugateCheck(main_sequence, storingIndex)) {
                        return new PatternMatchIndices(actualSentLevelElementIndices, entityIndices);
                    } else {
                        startingIndex = storingIndex.get(0) + 1;
                        storingIndex.clear();
                        patternTree.clear();
                        entityIndices.clear();
                        actualSentLevelElementIndices.clear();
                    }
                } else {
                    return new PatternMatchIndices(actualSentLevelElementIndices, entityIndices);
                }
            } else {
                return null;
            }
        }
    }

    public static boolean isSplitPoint(IndexedWord word, String[] nerTypes) {
        return word.tag().charAt(0) == 'N' || containsEntity(word.value(), nerTypes);
    }

    public static boolean isModifier(SemanticGraphEdge edge) {
        return edge != null && (edge.toString().contains("mod") || edge.toString().contains("compound"));
    }

    public static List<IndexedWord> verbAlternates(IndexedWord verb, SemanticGraph semanticGraph, String[] nerTypes,
                                                   Map<IndexedWord, TreeSet<IndexedWord>> subTree) {
        List<IndexedWord> potentialDeletions = new ArrayList<>();
        SVOClause mainClause = new SVOClause(verb);
        for (IndexedWord child : semanticGraph.getChildList(verb)) {
            if (child.index() < verb.index() && semanticGraph.getEdge(verb, child).toString().contains("subj")) {
                mainClause.setSubject(child);
            } else if (mainClause.foundSubject() && isSplitPoint(child, nerTypes) && child.index() > verb.index()) {
                mainClause.setObject(child);
            } else if (semanticGraph.getEdge(verb, child).toString().contains("cc") && child.index() > verb.index()) {
                potentialDeletions.add(child);
            } else if (semanticGraph.getEdge(verb, child).toString().contains("conj") && child.index() > verb.index()) {
                mainClause.setConjunct(child);
                potentialDeletions.add(child);
            }
            subTree.get(verb).addAll(Collections.singletonList(child));
        }
        if (mainClause.isCoordClause()) {
            TreeSet<IndexedWord> temp = subTree.getOrDefault(mainClause.getConjunct(), new TreeSet<>());
            SVOClause conjugateClause = returnClauseComponents(mainClause.getConjunct(), semanticGraph);
            if (!conjugateClause.foundSubject()) {
                temp.add(mainClause.getSubject());
            }
            if (!conjugateClause.foundObject() && mainClause.foundObject()
                    && mainClause.getObject().index() > mainClause.getConjunct().index()) {
                temp.add(mainClause.getObject());
            }
            else if (!mainClause.foundObject() && conjugateClause.foundObject()) {
                subTree.get(verb).addAll(Collections.singletonList(conjugateClause.getObject()));
            }
            subTree.put(mainClause.getConjunct(), temp);
            return potentialDeletions;
        }
        return null;
    }

    public static List<SubSentWords> sort_leafToTop(SentenceProcessor sentence){
        List<SubSentWords> l = new ArrayList<>(sentence.getSentenceBreakdown().keySet());
        // Arrange sub-roots from leaf to root of sentence
        l.sort((o1, o2) -> {
            Integer i2 = o2.getTrimmedEncoding().length();
            Integer i1 = o1.getTrimmedEncoding().length();
            return i2.compareTo(i1);
        });
        return l;
    }

    public static List<IndexedWord> sort_topToLeaf(TreeSet<IndexedWord> nodes, Map<IndexedWord, String> encodeTree) {
        return nodes.stream().sorted((a,b) -> {
            Integer a1 = encodeTree.get(a).split("_")[0].length();
            Integer a2 = encodeTree.get(b).split("_")[0].length();
            return a1.compareTo(a2);
        }).collect(Collectors.toList());
    }

    public static String returnLowercaseLemma(IndexedWord word) {
        if (word.lemma() == null) word.setLemma(Morphology.lemmaStatic(word.value(), word.tag()));
        return word.lemma().toLowerCase();
    }

    public static boolean containsEntity(String subRoot, String[] nerTypes) {
        return Arrays.stream(nerTypes).anyMatch(subRoot::contains);
    }

    public static int returnEncodeIndex(List<SubSentWords> encodeList, String encoding) {
        for (int i = 0; i < encodeList.size(); i++) {
            if (encodeList.get(i).getTrimmedEncoding().equals(encoding)) {
                return i;
            }
        }
        return -1;
    }

    public static void updateMapCount(Map<String, List<Integer>> map, String token, Integer value) {
        List<Integer> temp = map.getOrDefault(token, new ArrayList<>());
        temp.add(value);
        map.put(token, temp);
    }

    public static void makingDir(String folderName) {
        File dir = new File(folderName);
        if (!dir.exists()) {
            dir.mkdir();
        }
    }

    public static int countSavedBatches(String directory, int noOfLines) {
        int batchIterNo = 0;
        boolean breakdownDataExists;
        while (true) {
            breakdownDataExists = new File(directory + "sentenceBatch" + batchIterNo + "." + noOfLines + ".txt").exists();
            if (!breakdownDataExists) break;
            batchIterNo++;
        }
        return batchIterNo;
    }

    public static void writePatternsToFile(BufferedWriter bw, Map<String, List<MetaPattern>> patternList) throws IOException {
        Map<String, Integer> patternCount = new HashMap<>();
        patternList.forEach((key, val) -> patternCount.put(key, frequencySumHelper(val)));
        for (Map.Entry<String, Integer> metaPattern : sortDecreasing(patternCount)) {
            bw.write(metaPattern.getKey() + "->" + metaPattern.getValue() + "\n");
        }
        bw.close();
    }

    private static int frequencySumHelper(List<MetaPattern> patterns) {
        return patterns.stream().mapToInt(MetaPattern::getFrequency).sum();
    }

    public static void setOptionalParameterDefaults(Properties prop) {
        checkAndSetParameter (prop, "inputFolder", "Data/" + prop.getProperty("data_name") + "/");
        checkAndSetParameter (prop, "outputFolder", folderNameConsistency(prop.getProperty("inputFolder")) + "Output/");
        checkAndSetParameter (prop, "load_sentenceBreakdownData", "false");
        checkAndSetParameter (prop, "load_metapatternData", "false");
        checkAndSetParameter (prop, "minimumSupport", "5");
        checkAndSetParameter (prop, "maxLength", "15");
        checkAndSetParameter (prop, "stopWordsFile", "stopWords.txt");
        checkAndSetParameter (prop, "nerTypes", folderNameConsistency(prop.getProperty("inputFolder")) + "entityTypes.txt");
        checkAndSetParameter(prop, "noOfPushUps", "3");
        checkAndSetParameter(prop, "includeContext", "true");
        checkAndSetParameter(prop, "batchSize", "200000");
        checkAndSetParameter(prop, "threadCount", String.valueOf(Runtime.getRuntime().availableProcessors()));
    }

    public static List<String> readList(FileReader fr) throws IOException {
        BufferedReader br = new BufferedReader(fr);
        List<String> ans = new ArrayList<>();
        String line = br.readLine();
        while (line!= null) {
            ans.add(line);
            line = br.readLine();
        }
        br.close();
        return ans;
    }

    public static Map<String, Integer> returnSortedPatternList(Map<String, List<MetaPattern>> patternList) {
        Map<String, Integer> ans = new LinkedHashMap<>();
        List<String> l = new ArrayList<>(patternList.keySet());
        l.sort((o1, o2) ->
                returnPatternWeight(o2, patternList.get(o2)).compareTo(returnPatternWeight(o1, patternList.get(o1))));
        l.forEach(s -> ans.put(s, maxNerCount(patternList.get(s))));
        return ans;
    }

    public static String folderNameConsistency(String folderName) {
        String folder = folderName;
        if (folder.charAt(folder.length() - 1) != '/') {
            folder += "/";
        }
        return folder;
    }

    public static String trimEncoding(String encoding) {
        return encoding.split("_")[0];
    }

    public static int noOfEntities(List<SubSentWords> subSent, String[] nerTypes) {
        return (int) subSent.stream().map(SubSentWords::getLemma).filter(word -> Arrays.stream(nerTypes).anyMatch(word::contains)).count();
    }

    public static String createExtractionString(List<PatternInstance> patternOutputs) {
        Set<Integer> isSingleEntityPattern = new HashSet<>();
        Map<Integer, Set<Integer>> finalContextMapping = new HashMap<>();
        Map<SubSentWords, List<Integer>> wordWisePatterns = new HashMap<>();

        for (int i = 0; i < patternOutputs.size(); i++) {
            List<SubSentWords> entityList = patternOutputs.get(i).getEntities();
            if (entityList.size() > 1) finalContextMapping.put(i, new HashSet<>());
            else isSingleEntityPattern.add(i);
            int finalI = i;
            entityList.forEach(entity -> {
                List<Integer> temp = wordWisePatterns.getOrDefault(entity, new ArrayList<>());
                temp.add(finalI);
                wordWisePatterns.put(entity, temp);
            });
        }
        for (List<Integer> values : wordWisePatterns.values()) {
            Set<Integer> mains = values.stream().filter(s -> !isSingleEntityPattern.contains(s)).collect(Collectors.toSet());
            Set<Integer> contexts = values.stream().filter(isSingleEntityPattern::contains).collect(Collectors.toSet());
            if (!mains.isEmpty()) {
                for (Integer i : mains) {
                    finalContextMapping.get(i).addAll(contexts);
                }
            } else {
                int maxVal = contexts.stream().max(Comparator.comparingInt(s -> patternOutputs.get(s).getAllElementIndices().size())).orElse(-1);
                if (maxVal != -1) {
                    finalContextMapping.put(maxVal, finalContextMapping.getOrDefault(maxVal, new HashSet<>()));
                    for (Integer i : contexts) {
                        if (i != maxVal) finalContextMapping.get(maxVal).add(i);
                    }
                }
            }
        }
        StringBuilder ans = new StringBuilder();
        for (Integer i : finalContextMapping.keySet()) {
            ans.append(patternOutputs.get(i).toString());
            if (finalContextMapping.get(i).size() != 0) {
                ans.append("{Context:\n");
                finalContextMapping.get(i).forEach(s -> ans.append(patternOutputs.get(s).toString()));
                ans.append("}\n");
            }
        }
        return ans.toString();
    }

    private static Double returnPatternWeight(String metaPattern, List<MetaPattern> patternList) {
        return (double) maxNerCount(patternList) + metaPattern.replace("_", " ").split(" ").length/20.0;
    }

    private static boolean noConjugateCheck(List<SubSentWords> mainSequence, List<Integer> storingIndex) {
        List<String> encoding = storingIndex.stream().map(s -> mainSequence.get(s).getEncoding()).collect(Collectors.toList());
        String parent = null;
        for (String e : encoding) {
            if (e.contains("conj")) {
                parent = e.split("_")[0];
                parent = parent.substring(0, parent.length() - 1);
                break;
            }
        }
        if (parent == null) return true;
        for (String e : encoding) {
            if (e.split("_")[0].equals(parent)) {
                return false;
            }
        }
        return true;
    }

    private static List<String> returnEntitiesFromPattern(String patternOutput) {
        String entities = patternOutput.split("\t")[2];
        String[] entityList = String.join(",", entities.substring(1, entities.length() - 1).split(" , ")).split(", ");
        return Arrays.asList(entityList);
    }

    private static SVOClause returnClauseComponents(IndexedWord parent, SemanticGraph semanticGraph) {
        SVOClause answer = new SVOClause(parent);
        for (IndexedWord child : semanticGraph.getChildList(parent)) {
            if (semanticGraph.getEdge(parent, child).toString().contains("subj") && child.index() < parent.index()) {
                answer.setSubject(child);
            }
            else if (child.tag().charAt(0) == 'N' && child.index() > parent.index()) {
                answer.setObject(child);
            }
        }
        return answer;
    }

    private static void checkAndSetParameter(Properties prop, String key, String value) {
        if (!prop.containsKey(key) || prop.getProperty(key).equals("")) {
            prop.setProperty(key, value);
        }
    }

    private static Integer maxNerCount(List<MetaPattern> patterns) {
        if (patterns.isEmpty()) return 0;
        int max = Integer.MIN_VALUE;
        for (MetaPattern p : patterns) {
            int temp = p.getNerCount();
            if (temp > max) max = temp;
        }
        return max;
    }
}
