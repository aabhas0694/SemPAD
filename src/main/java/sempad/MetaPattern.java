package sempad;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NonNull;

import java.util.*;

@Getter
@AllArgsConstructor
public class MetaPattern {
    @NonNull
    private final String[] splitPattern;
    private Set<String> entityContainingWords = new HashSet<>();
    private List<Integer> entityIndex = new ArrayList<>();

    public MetaPattern(String[] splitPattern) {
        this.splitPattern = splitPattern;
    }

    public int getNerCount() {
        return entityIndex.size();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        MetaPattern that = (MetaPattern) o;
        return Arrays.equals(splitPattern, that.splitPattern);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(splitPattern);
    }

    @Override
    public String toString() {
        return String.join(" ", this.splitPattern);
    }
}
