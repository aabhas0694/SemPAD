package cpww;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import static cpww.utils.Util.containsEntity;
import static cpww.utils.Util.trimEncoding;

public class PatternInstance {
    private String sentID;
    private String metaPattern;
    private List<String> entities = new ArrayList<>();
    private List<List<String>> alternateEntities = new ArrayList<>();
    private StringBuilder sentenceInstance;

    PatternInstance (SentenceProcessor sentence, SubSentWords subRoot, String metaPattern, List<Integer> entityPos,
                     String[] nerTypes) {
        this.metaPattern = metaPattern;
        this.sentID = sentence.getSentID();
        Map<String, SubSentWords> encodingMap = sentence.getReverseWordEncoding();
        List<SubSentWords> value = sentence.getSentenceBreakdown().get(subRoot);
        List<String> subEnc = value.stream().map(SubSentWords::getEncoding).collect(Collectors.toList());
        String encoding = String.join(" ", subEnc);
        for (Integer entityPo : entityPos) {
            String entity_code = subEnc.get(entityPo);
            if (sentence.getReplaceSurfaceName().containsKey(trimEncoding(entity_code)) &&
                    !trimEncoding(entity_code).equals(subRoot.getTrimmedEncoding())) {
                List<String> output = hierarchical_expansion(sentence, encoding, entity_code);
                this.entities.add(encodingMap.get(output.get(0)).getOriginalWord());
                this.alternateEntities.add(conjunctionSearch(output.get(0), sentence));
                encoding = output.get(1);
            } else {
                this.entities.add(encodingMap.get(entity_code).getOriginalWord());
                this.alternateEntities.add(conjunctionSearch(entity_code, sentence));
            }
        }
        sentenceInstance = new StringBuilder();
        sentenceInstance.append(map_original_tokens(sentence, encoding, nerTypes, false)).append("\t");
        sentenceInstance.append(map_original_tokens(sentence, encoding, nerTypes, true));
    }

    private static List<String> conjunctionSearch(String encoding, SentenceProcessor sentence) {
        SubSentWords subRoot = sentence.getReverseWordEncoding().get(encoding);
        String encode = encoding.split("_")[0].replaceAll("\\[", "\\\\[");
        List<SubSentWords> entitySent = sentence.getSentenceBreakdown().get(subRoot);
        List<String> ans = new ArrayList<>();
        String ccPattern = "^" + encode + "[\\w]_cc";
        if (entitySent.parallelStream().anyMatch(sw -> Pattern.matches(ccPattern, sw.getEncoding())
                && (sw.getLemma().equals("but") || sw.getLemma().equals("between")))) {
            return null;
        }
        if (subRoot.getEncoding().contains("_conj")) {
            SubSentWords ancestor = sentence.getReverseWordEncoding().getOrDefault(encode.substring(0, encode.length() - 1), null);
            if (ancestor != null && ancestor.getLemma().equals(subRoot.getLemma())){
                ans.add(ancestor.getOriginalWord());
                encode = ancestor.getTrimmedEncoding();
                entitySent = sentence.getSentenceBreakdown().get(ancestor);
            }
        }
        String conjPattern = "^" + encode + "[\\w]_conj";
        for (SubSentWords sw : entitySent) {
            if (!sw.equals(subRoot) && Pattern.matches(conjPattern, sw.getEncoding()) && sw.getLemma().equals(subRoot.getLemma()) &&
            sentence.getSentenceBreakdown().get(sw).size() == 1) {
                ans.add(sw.getOriginalWord());
            }
        }
        return ans.isEmpty() ? null : ans;
    }

    private static List<String> hierarchical_expansion(SentenceProcessor sentence, String original_subEnc,
                                                       String entityEncode){
        List<SubSentWords> toBeAddedSubSent = sentence.getSentenceBreakdown().get(sentence.getReverseWordEncoding().get(entityEncode));
        String added_subEnc = toBeAddedSubSent.stream().map(SubSentWords::getEncoding).collect(Collectors.joining(" "));
        SubSentWords replace_encode = sentence.getReplaceSurfaceName().get(trimEncoding(entityEncode));
        int index_diff = original_subEnc.length() - entityEncode.length();

        List<String> ans = new ArrayList<>();
        ans.add(replace_encode.getEncoding());
        if (Arrays.asList(original_subEnc.replaceAll("[{}]", "").split(" ")).contains(replace_encode.getEncoding())
                || replace_encode.getEncoding().equals(entityEncode)) {
            // Do nothing
        } else if (original_subEnc.contains(" " + entityEncode + " ")) {
            original_subEnc = original_subEnc.replace(" " + entityEncode + " ", " {{" + added_subEnc + "}} ");
        } else if (original_subEnc.contains("{{" + entityEncode + " ")) {
            original_subEnc = original_subEnc.replace("{{" + entityEncode + " ", "{{{{" + added_subEnc + "}} ");
        } else if (original_subEnc.contains(" " + entityEncode + "}}")) {
            original_subEnc = original_subEnc.replace(" " + entityEncode + "}}", " {{" + added_subEnc + "}}}}");
        } else if (original_subEnc.contains("{{" + entityEncode + "}}")) {
            original_subEnc = original_subEnc.replace("{{" + entityEncode + "}}", "{{{{" + added_subEnc + "}}}}");
        } else if (index_diff > 0 && original_subEnc.substring(index_diff - 1).contains(" " + entityEncode)) {
            original_subEnc = original_subEnc.substring(0, index_diff - 1) + " {{" + added_subEnc + "}}";
        } else if (original_subEnc.contains(entityEncode + " ")) {
            original_subEnc = "{{" + added_subEnc + "}} " + original_subEnc.substring(entityEncode.length() + 1);
        }
        if (!sentence.getReplaceSurfaceName().containsKey(replace_encode.getTrimmedEncoding()) ||
                replace_encode.getEncoding().equals(entityEncode)) {
            ans.add(original_subEnc);
            return ans;
        }
        return hierarchical_expansion(sentence, original_subEnc, replace_encode.getEncoding());
    }

    private String map_original_tokens(SentenceProcessor sentence, String encode, String[] nerTypes, boolean wantLemma){
        Map<String, SubSentWords> encodingMap = sentence.getReverseWordEncoding();
        String[] enc = encode.split(" ");
        List<String> ans = new ArrayList<>();
        for (String encoding : enc){
            StringBuilder temp = new StringBuilder(), res = new StringBuilder(), finalRes = new StringBuilder();
            boolean flag = false;
            for (int i = 0; i < encoding.length(); i++){
                if (res.length() == 0 && encoding.charAt(i) == '{'){
                    temp.append("{");
                }
                else if(res.length() != 0 && encoding.charAt(i) == '}' && !flag){
                    SubSentWords word = encodingMap.get(res.toString());
                    finalRes.append(temp);
                    if (wantLemma && !containsEntity(word.getLemma(), nerTypes)) {
                        finalRes.append(word.getLemma()).append("}");
                    } else {
                        finalRes.append(word.getOriginalWord()).append("}");
                    }
                    flag = true;
                }
                else if (res.length() != 0 && encoding.charAt(i) == '}' && flag) finalRes.append("}");
                else res.append(encoding.charAt(i));
            }
            if (!flag) {
                SubSentWords word = encodingMap.get(res.toString());
                finalRes.append(temp);
                if (wantLemma && !containsEntity(word.getLemma(), nerTypes)) {
                    finalRes.append(word.getLemma());
                } else {
                    finalRes.append(word.getOriginalWord());
                }
            }
            ans.add(finalRes.toString());
        }
        return String.join(" ", ans);
    }

    public String toString() {
        if (metaPattern == null) return null;
        StringBuilder output = new StringBuilder(sentID);
        output.append("\t").append(metaPattern).append("\t[");
        output.append(String.join(", ", entities)).append("]\t").append(sentenceInstance).append("\n");
        if (!this.alternateEntities.isEmpty()) {
            for (int i = 0; i < alternateEntities.size(); i++) {
                if (alternateEntities.get(i) != null) {
                    for (int j = 0; j < alternateEntities.get(i).size(); j++) {
                        output.append(sentID).append("\t").append(metaPattern).append("\t[");
                        List<String> temp = new ArrayList<>(entities.subList(0, i));
                        temp.add(alternateEntities.get(i).get(j));
                        if (i + 1 != entities.size()) {
                            temp.addAll(entities.subList(i + 1, entities.size()));
                        }
                        output.append(String.join(", ", temp)).append("]\t")
                                .append(sentenceInstance.toString().replace(entities.get(i), alternateEntities.get(i).get(j)))
                                .append("\n");
                    }
                }
            }
        }
        return output.toString();
    }
}
