package cpww;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class PatternInstance {
    private String sentID;
    private String metaPattern;
    private List<String> entities;
    private List<List<String>> alternateEntities;
    private String sentenceInstance;

    PatternInstance (SentenceProcessor sentence, SubSentWords subRoot, String metaPattern, List<Integer> entityPos) {
        this.metaPattern = metaPattern;
        this.sentID = sentence.getSentID();
        Map<String, SubSentWords> encodingMap = sentence.getReverseWordEncoding();
        int adjust_length = 0;
        List<SubSentWords> value = sentence.getSentenceBreakdown().get(subRoot);
        List<String> subEnc = value.stream().map(SubSentWords::getEncoding).collect(Collectors.toList());
        String encoding = String.join(" ", subEnc);
        for (Integer entityPo : entityPos) {
            SubSentWords entity_code = value.get(entityPo + adjust_length);
            if (sentence.getReplaceSurfaceName().containsKey(entity_code.getTrimmedEncoding()) &&
                    !entity_code.getTrimmedEncoding().equals(subRoot.getTrimmedEncoding())) {
                List<String> output = hierarchical_expansion(sentence, encoding, entity_code);
                this.entities.add(encodingMap.get(output.get(0)).getOriginalWord());
                List<SubSentWords> entitySent = sentence.getSentenceBreakdown().get(encodingMap.get(output.get(0)));
                alternateEntities.add(conjunctionSearch(output.get(0), entitySent));
                encoding = output.get(1);
            } else {
                entities.add(entity_code.getOriginalWord());
                List<SubSentWords> entitySent = sentence.getSentenceBreakdown().get(entity_code);
                alternateEntities.add(conjunctionSearch(entity_code.getEncoding(), entitySent));
            }
            sentenceInstance = map_original_tokens(sentence, encoding);
        }
    }

    private static List<String> conjunctionSearch(String encoding, List<SubSentWords> subSent) {
        List<String> ans = new ArrayList<>();
        String pattern = "^" + encoding.split("_")[0] + "[\\s]_conj";
        for (SubSentWords sw : subSent) {
            if (Pattern.matches(pattern, sw.getEncoding())) {
                ans.add(sw.getOriginalWord());
            }
        }
        return ans.isEmpty() ? null : ans;
    }

    private static List<String> hierarchical_expansion(SentenceProcessor sentence, String original_subEnc,
                                                       SubSentWords entityEncode) {
        List<SubSentWords> toBeAddedSubSent = sentence.getSentenceBreakdown().get(entityEncode);
        String added_subEnc = toBeAddedSubSent.stream().map(SubSentWords::getEncoding).collect(Collectors.joining(" "));
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
                replace_encode.getEncoding().equals(entityEncode.getEncoding())) {
            ans.add(original_subEnc);
            return ans;
        }
        return hierarchical_expansion(sentence, original_subEnc, replace_encode);
    }

    private String map_original_tokens(SentenceProcessor sentence, String encode){
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

    public String toString() {
        if (metaPattern == null) return null;
        String start = sentID + "\t" + metaPattern + "\t[";
        String output = start;
        for (String entity : entities) {
            output += entity + ", ";
        }
        output = output.substring(0, output.length() - 2) + "]\t" + sentenceInstance + "\n";
        if (alternateEntities != null) {
            for (int i = 0; i < alternateEntities.size(); i++) {
                if (alternateEntities.get(i) != null) {
                    for (int j = 0; j < alternateEntities.get(i).size(); j++) {
                        output += start;
                        List<String> temp = new ArrayList<>();
                        temp.addAll(entities.subList(0, i));
                        temp.add(alternateEntities.get(i).get(j));
                        if (i + 1 != entities.size()) {
                            temp.addAll(entities.subList(i + 1, entities.size()));
                        }
                        output += String.join(", ", temp) + "]\tConjunctions Found\n";
                    }
                }
            }
        }
        return output;
    }
}
