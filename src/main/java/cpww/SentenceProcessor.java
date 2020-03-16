package cpww;

import java.io.Serializable;
import java.util.*;

import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.BasicDependenciesAnnotation;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.semgraph.SemanticGraph;

public class SentenceProcessor implements Serializable {
    private String sentence;
    private String sentID;
    private Map<SubSentWords, List<SubSentWords>> sentenceBreakdown = new LinkedHashMap<>();
    private Map<String, SubSentWords> reverseEncoding = new HashMap<>();
    private Map<String, SubSentWords> replaceSurfaceName = new HashMap<>();
    private Map<SubSentWords, List<SubSentWords>> pushedUpSentences = new HashMap<>();

    public SentenceProcessor(StanfordCoreNLP pipeline, String[] nerTypes, Map<String, String> entityDictionary,
                             String sentence, String sentID) {
        super();
        processSentence(sentence, sentID, pipeline, entityDictionary, nerTypes);
    }

    public void processSentence(String text, String sentID, StanfordCoreNLP pipeline, Map<String, String> entityDict, String[] nerTypes) {
        this.sentence = text;
        this.sentID = sentID;
        SemanticGraph semanticGraph = generateSemanticGraph(text, pipeline);
        Map<IndexedWord, String> encodeTree = encodeTree(semanticGraph, semanticGraph.getFirstRoot(), "A_root", new HashMap<>());
        generateReverseWordEncoding(encodeTree, entityDict, nerTypes);
        Map<IndexedWord, TreeSet<IndexedWord>> subTree = splitNoun(semanticGraph.getFirstRoot(), new HashMap<>(),  semanticGraph, nerTypes);
        sentenceBreakdown(encodeTree, subTree);
    }

    private SemanticGraph generateSemanticGraph(String text, StanfordCoreNLP pipeline) {
        Annotation ann = new Annotation(text);
        pipeline.annotate(ann);
        CoreMap sentence = ann.get(SentencesAnnotation.class).get(0);
        return sentence.get(BasicDependenciesAnnotation.class);
    }

    private Map<IndexedWord, TreeSet<IndexedWord>> splitNoun(IndexedWord root, Map<IndexedWord,
            TreeSet<IndexedWord>> subTree, SemanticGraph semanticGraph, String[] nerTypes) {
        if (subTree.containsKey(root)) {
            return subTree;
        }
        int index = root.index();
        subTree.put(root, new TreeSet<>());
        Queue<IndexedWord> search = new LinkedList<>();
        search.offer(root);
        while (!search.isEmpty()) {
            IndexedWord node = search.poll();
            if (!subTree.get(root).contains(node)) {
                TreeSet<IndexedWord> temp = subTree.get(root);
                temp.add(node);
                subTree.put(root, temp);
            }
            int newIndex = node.index();
            boolean rootCheck = semanticGraph.getRoots().contains(node);
            List<IndexedWord> children = semanticGraph.getChildList(node);

            for (IndexedWord child : children) {
                if (rootCheck || !isSplitPoint(node, nerTypes)) {
                    search.offer(child);
                } else {
                    if (root.equals(node)) {
                        if (child.index() >= index) search.offer(child);
                    } else {
                        if (child.index() <= newIndex) search.offer(child);
                    }
                }
            }
            if (isSplitPoint(node, nerTypes)) {
                subTree = splitNoun(node, subTree, semanticGraph, nerTypes);
            }
        }
        return subTree;
    }

    private Map<IndexedWord, String> encodeTree(SemanticGraph semanticGraph, IndexedWord r, String coding, Map<IndexedWord, String> map) {
        if (map.containsKey(r)) return map;
        map.put(r, coding);
        int i = 0;
        for (SemanticGraphEdge edge : semanticGraph.getOutEdgesSorted(r)) {
            char ch = (char) ((int) 'A' + i++);
            String new_coding = coding.split("_")[0] + ch + '_' + edge.getRelation();
            map = encodeTree(semanticGraph, edge.getDependent(), new_coding, map);
        }
        return map;
    }

    private boolean containsEntity(String subRoot, String[] nerTypes) {
        for (String ner : nerTypes) {
            if (subRoot.contains(ner)) {
                return true;
            }
        }
        return false;
    }

    private void generateReverseWordEncoding(Map<IndexedWord, String> encodeTree, Map<String, String> entityDict, String[] nerTypes) {
        for(Map.Entry<IndexedWord, String> entry : encodeTree.entrySet()){
            SubSentWords word = new SubSentWords(entry.getKey(), entry.getValue(),
                    this.containsEntity(entry.getKey().value(), nerTypes));
            word.setWord(entityDict.getOrDefault(entry.getKey().value(), entry.getKey().value()));
            this.reverseEncoding.put(entry.getValue(), word);
        }
    }

    private boolean isSplitPoint(IndexedWord word, String[] nerTypes) {
        return word.tag().charAt(0) == 'N' || containsEntity(word.value(), nerTypes);
    }

    public Map<String, SubSentWords> getReverseWordEncoding() {
        return this.reverseEncoding;
    }

    private void sentenceBreakdown(Map<IndexedWord, String> encodeTree, Map<IndexedWord, TreeSet<IndexedWord>> subTree) {
        for (IndexedWord key : subTree.keySet()) {
            SubSentWords subRoot = this.reverseEncoding.get(encodeTree.get(key));
            TreeSet<IndexedWord> values = subTree.get(key);
            List<SubSentWords> temp = new ArrayList<>();
            for (IndexedWord w : values) {
                SubSentWords tempWord = this.reverseEncoding.get(encodeTree.get(w));
                temp.add(tempWord);
            }
            this.sentenceBreakdown.put(subRoot, temp);
        }
    }

    public Map<SubSentWords, List<SubSentWords>> getSentenceBreakdown() {
        return this.sentenceBreakdown;
    }

    public void pushUpSentences(SubSentWords encode, List<SubSentWords> value) {
        this.pushedUpSentences.put(encode, value);
    }

    public Map<SubSentWords, List<SubSentWords>> getPushedUpSentences() {
        return this.pushedUpSentences;
    }

    public Map<String, SubSentWords> getReplaceSurfaceName() {
        return this.replaceSurfaceName;
    }
    public void updateReplacedSurfaceName(String key, SubSentWords value) {
        this.replaceSurfaceName.put(key, value);
    }

    public void resetPushUp() {
        this.replaceSurfaceName.clear();
        this.pushedUpSentences.clear();
    }

    public String getSentID() {
        return this.sentID;
    }
}
