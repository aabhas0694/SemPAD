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
    private String[] nerTypes;
    private SemanticGraph semanticGraph;
    private Map<IndexedWord, TreeSet<IndexedWord>> subTree = new HashMap<>();
    private Map<SubSentWords, List<SubSentWords>> sentenceBreakdown = new LinkedHashMap<>();
    private List<IndexedWord> roots = new ArrayList<>();
    private Map<IndexedWord, String> encodeTree = new HashMap<>();

    private Map<String, SubSentWords> replaceSurfaceName = new HashMap<>();
    private Map<SubSentWords, List<SubSentWords>> pushedUpSentences = new HashMap<>();

    private Map<String, String> entityDictionary;

    public SentenceProcessor(StanfordCoreNLP pipeline, String[] nerTypes, Map<String, String> entityDictionary,
                             String sentence, String sentID) {
        super();
        this.nerTypes = nerTypes;
        this.entityDictionary = entityDictionary;
        processSentence(sentence, sentID, pipeline);
    }

    public void processSentence(String text, String sentID, StanfordCoreNLP pipeline) {
        this.sentence = text;
        this.sentID = sentID;
        setParameters(text, pipeline);
        encodeTree(this.semanticGraph.getFirstRoot(), "A_root");
        splitNoun(this.semanticGraph.getFirstRoot());
        sentenceBreakdown();
    }

    private void setParameters(String text, StanfordCoreNLP pipeline) {
        Annotation ann = new Annotation(text);
        pipeline.annotate(ann);
        CoreMap sentence = ann.get(SentencesAnnotation.class).get(0);
        this.semanticGraph = sentence.get(BasicDependenciesAnnotation.class);
        this.roots = new ArrayList<>(this.semanticGraph.getRoots());
    }

    public List<IndexedWord> getRoots() {
        return this.roots;
    }

    public SemanticGraph getSemanticGraph() {
        return this.semanticGraph;
    }

    private void splitNoun(IndexedWord root) {
        if (this.subTree.containsKey(root)) {
            return;
        }
        int index = root.index();
        this.subTree.put(root, new TreeSet<>());
        Queue<IndexedWord> search = new LinkedList<>();
        search.offer(root);
        while (!search.isEmpty()) {
            IndexedWord node = search.poll();
            if (!this.subTree.get(root).contains(node)) {
                TreeSet<IndexedWord> temp = this.subTree.get(root);
                temp.add(node);
                this.subTree.put(root, temp);
            }
            int newIndex = node.index();
            boolean rootCheck = this.roots.contains(node);
            List<IndexedWord> children = this.semanticGraph.getChildList(node);

            for (IndexedWord child : children) {
                if (rootCheck || node.tag().charAt(0) != 'N') {
                    search.offer(child);
                } else {
                    if (root.equals(node)) {
                        if (child.index() >= index) search.offer(child);
                    } else {
                        if (child.index() <= newIndex) search.offer(child);
                    }
                }
            }
            if (node.tag().charAt(0) == 'N') {
                splitNoun(node);
            }
        }
    }

    private void encodeTree(IndexedWord r, String coding) {
        if (this.encodeTree.containsKey(r)) return;
        this.encodeTree.put(r, coding);
        int i = 0;
        for (SemanticGraphEdge edge : semanticGraph.getOutEdgesSorted(r)) {
            char ch = (char) ((int) 'A' + i++);
            String new_coding = coding.split("_")[0] + ch + '_' + edge.getRelation();
            encodeTree(edge.getDependent(), new_coding);
        }
    }

    private boolean containsEntity(String subRoot) {
        for (String ner : this.nerTypes) {
            if (subRoot.contains(ner)) {
                return true;
            }
        }
        return false;
    }

    public Map<String, SubSentWords> getReverseWordEncoding() {
        Map<String, SubSentWords> myNewHashMap = new HashMap<>();
        for(Map.Entry<IndexedWord, String> entry : this.encodeTree.entrySet()){
            SubSentWords word = new SubSentWords(entry.getKey(), entry.getValue(),
                    this.containsEntity(entry.getKey().value()));
            word.setWord(entityDictionary.getOrDefault(entry.getKey().value(), entry.getKey().value()));
            myNewHashMap.put(entry.getValue(), word);
        }
        return myNewHashMap;
    }

    private void sentenceBreakdown() {
        for (IndexedWord key : this.subTree.keySet()) {
            SubSentWords subRoot = new SubSentWords(key, this.encodeTree.get(key), this.containsEntity(key.value()));
            subRoot.setWord(entityDictionary.getOrDefault(key.value(), key.value()));
            TreeSet<IndexedWord> values = this.subTree.get(key);
            List<SubSentWords> temp = new ArrayList<>();
            for (IndexedWord w : values) {
                SubSentWords tempWord = new SubSentWords(w, this.encodeTree.get(w), this.containsEntity(w.value()));
                tempWord.setWord(entityDictionary.getOrDefault(w.value(), w.value()));
                temp.add(tempWord);
            }
            this.sentenceBreakdown.put(subRoot, temp);
        }
    }

//    public String returnSentenceBreakdown(String mode) throws Exception {
//        if (this.sentenceBreakdown.isEmpty()) {
//            throw new Exception("No Breakdown Data Exists.");
//        }
//        StringBuilder result = new StringBuilder();
//        result.append(this.sentID).append("\n");
//        Map<String, List<String>> toPrint = new HashMap<>();
//        toPrint = mode.toLowerCase().equals("subsent") ? this.subSentence : this.subEncode;
//        for (String key : toPrint.keySet()) {
//            result.append(key).append("->").append(String.join(" ", toPrint.get(key))).append("\n");
//        }
//        return result.toString();
//    }

//    public void printSentenceBreakdown(String mode) throws Exception {
//        System.out.print(this.returnSentenceBreakdown(mode));
//    }

//    public void setSentenceProcess(String sentID, List<String> sent, List<String> enc) throws IOException {
//        if (sent.size() != enc.size()) {
//            throw new IOException("Sentence and Encoding size do not match.");
//        }
//        this.sentID = sentID;
//        for (int i = 0; i < sent.size(); i++) {
//            String[] sentBreak = sent.get(i).split("->");
//            String[] encBreak = enc.get(i).split("->");
//            this.subSentence.put(sentBreak[0], Arrays.asList(sentBreak[1].split(" ")));
//            this.subEncode.put(encBreak[0], Arrays.asList(encBreak[1].split(" ")));
//        }
//    }

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
