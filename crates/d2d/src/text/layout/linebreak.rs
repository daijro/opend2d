//! Line-break opportunities as DirectWrite finds them: UAX #14's pair-table algorithm (the original example
//! implementation's direct, indirect, combining and prohibited actions) over 40 classes, with DirectWrite's own pair
//! table. It predates later UAX #14 rules, so it breaks a letter from a following prefix or postfix (between
//! 'Public' and '\Documents' in a path: VM layout clusters), and it has no emoji base, modifier or joiner classes.

use unicode_linebreak::{break_property, BreakClass, BreakOpportunity};

// The classes in alphabetical order, the table's index order.
const AL: u8 = 1;
const BA: u8 = 3;
const CB: u8 = 6;
const CM: u8 = 9;
const GL: u8 = 13;
const HL: u8 = 16;
const HY: u8 = 17;
const ID: u8 = 18;
const SP: u8 = 35;
const WJ: u8 = 37;

/// DirectWrite's action for each (before, after) class pair: 0 a direct break, 1 an indirect one (only after a space),
/// 2 a combining mark that takes the class before it, 3 one after an opening punctuation, 4 no break, 5 a mandatory
/// break. A letter marks the pairs the Hebrew-hyphen rule (LB21a) revisits, its action `letter - 'A'`.
const PAIRS: [&[u8; 40]; 40] = [
    b"1101040142444100110140004411100101144414", // AI
    b"1101040142444100110140004411100101144414", // AL
    b"0041040142444100010040004410000100044404", // B2
    b"AAABAEABECEEEAAAABAAEAAAEEBAAAABAAAEEEAE", // BA
    b"1111140142444111111141114411111111144414", // BB
    b"5555555555555555555555555555555555555555", // BK
    b"0000040042444100000040004400000100044404", // CB
    b"0001040142444100010040004410000100044404", // CJ
    b"0001040442444100010040004440011100044404", // CL
    b"1101040142444100110140004411100101144414", // CM
    b"1101040442444100110040004441011101144414", // CP
    b"5555555555555555555555554555555555555555", // CR
    b"0001040142444100010040004410000100044404", // EX
    b"1111141142444111111141114411111111144414", // GL
    b"0001040142444100010140004410010100044404", // H2
    b"0001040142444100010140004410010100044404", // H3
    b"110B0401424441001B0140004411100101144414", // HL
    b"AAABAEABECEEEAAAABAAEAAAEEBBAAABAAAEEEAE", // HY
    b"0001040142444100010140004410010100044404", // ID
    b"0001040142444100010140004410000100044404", // IN
    b"1101040142444100110040004411000101144414", // IS
    b"0001040142444100010140014410010100044404", // JL
    b"0001040142444100010140004410010100044404", // JT
    b"0001040142444100010140104410010100044404", // JV
    b"5555555555555555555555555555555555555555", // LF
    b"5555555555555555555555555555555555555555", // NL
    b"0001040142444100010040004410000100044404", // NS
    b"1101040142444100110140004411111101144414", // NU
    b"4444444443444444444444444444444444444444", // OP
    b"1101040142444100110040004411100101144414", // PO
    b"1101040142444111111041114411100101144414", // PR
    b"1111141142444111111141114411411111144414", // QU
    b"0001040142444100010040004410000110044404", // RI
    b"1101040142444100110140004411100101144414", // SA
    b"1101040142444100110140004411100101144414", // SG
    b"0000040040444000000040004400000000044404", // SP
    b"0001040142444100010040004411000100044404", // SY
    b"1111141142444111111141114411111111144414", // WJ
    b"1101040142444100110140004411100101144414", // XX
    b"0000040000040000000000004400000000040004", // ZW
];

fn action(before: u8, after: u8) -> u8 {
    let a = PAIRS[before as usize][after as usize];
    if a >= b'A' { 0x80 | (a - b'A') } else { a - b'0' }
}

/// A character's class: its Unicode line-break class, with the classes the table lacks folded in (an emoji base is
/// an ideograph, an emoji modifier a combining mark, a zero-width joiner glue) and U+FFFE and U+FFFF word joiners.
fn class(cp: u32) -> u8 {
    use BreakClass::*;
    if matches!(cp, 0xFFFE | 0xFFFF) {
        return WJ;
    }
    match break_property(cp) {
        Ambiguous => 0,
        Alphabetic => AL,
        BeforeAndAfter => 2,
        After => BA,
        Before => 4,
        Mandatory => 5,
        Contingent => CB,
        ConditionalJapaneseStarter => 7,
        ClosePunctuation => 8,
        CombiningMark | EmojiModifier => CM,
        CloseParenthesis => 10,
        CarriageReturn => 11,
        Exclamation => 12,
        NonBreakingGlue | ZeroWidthJoiner => GL,
        HangulLvSyllable => 14,
        HangulLvtSyllable => 15,
        HebrewLetter => HL,
        Hyphen => HY,
        Ideographic | EmojiBase => ID,
        Inseparable => 19,
        InfixSeparator => 20,
        HangulLJamo => 21,
        HangulTJamo => 22,
        HangulVJamo => 23,
        LineFeed => 24,
        NextLine => 25,
        NonStarter => 26,
        Numeric => 27,
        OpenPunctuation => 28,
        Postfix => 29,
        Prefix => 30,
        Quotation => 31,
        RegionalIndicator => 32,
        ComplexContext => 33,
        Surrogate => 34,
        Space => SP,
        Symbol => 36,
        WordJoiner => WJ,
        Unknown => 38,
        ZeroWidthSpace => 39,
    }
}

/// The pair walk's state: `cls` is the class the next pair looks up (spaces and absorbed combining marks leave it),
/// `prev` the previous character's own class, `prev2` the class `cls` held before it last changed.
struct Walk {
    cls: u8,
    prev: u8,
    prev2: u8,
}

impl Walk {
    fn new(first: u8) -> Self {
        Walk { cls: first, prev: first, prev2: 40 }
    }

    /// The break condition before the next character, of class `next`, as `DWRITE_BREAK_CONDITION`: 1 can break, 2 may
    /// not, 3 must. 0 keeps the condition an earlier pass set there (the table's indirect break with no space before).
    fn step(&mut self, next: u8) -> u8 {
        let mut act = action(self.cls, next);
        let mut update = next != SP;
        let mut cond = 2;
        let mut decided = false;
        if act & 0x80 != 0 {
            act &= 0x7f;
            // LB21a: no break after a Hebrew letter and a hyphen (or a break-after), unless a space parted them.
            if self.cls == HL {
                if (next == HY || next == BA) && self.prev == SP {
                    self.cls = SP;
                }
            } else if self.prev2 == HL && self.prev != SP && next != CB {
                update &= next != CM;
                decided = true;
            }
        }
        if !decided {
            match act {
                0 => cond = 1,
                1 => cond = if self.prev == SP { 1 } else { 0 },
                2 if self.prev != SP => {
                    // A combining mark takes the class before it.
                    self.prev = next;
                    return 2;
                }
                2 => cond = 1,
                3 => update &= self.prev == SP,
                4 => {}
                _ => {
                    (self.prev2, self.cls, self.prev) = (self.cls, next, next);
                    return 3;
                }
            }
        }
        if update {
            (self.prev2, self.cls) = (self.cls, next);
        }
        self.prev = next;
        cond
    }
}

/// The break opportunities of `text`, keyed by the byte offset after each break, as `unicode_linebreak::linebreaks`
/// reports them (the mandatory break at the end of the text included). `marked`: sorted offsets an earlier pass (the
/// Thai word breaker) marked breakable, which stand only where the table's indirect break finds no space before
/// (VM layout clusters: a Thai word end before a space does not break).
pub(super) fn linebreaks<'a>(text: &'a str, marked: &'a [usize]) -> impl Iterator<Item = (usize, BreakOpportunity)> + 'a {
    let mut chars = text.char_indices().map(|(i, c)| (i, class(c as u32)));
    let mut walk = chars.next().map(|(_, k)| Walk::new(k));
    chars
        .filter_map(move |(i, next)| match walk.as_mut()?.step(next) {
            0 if marked.binary_search(&i).is_ok() => Some((i, BreakOpportunity::Allowed)),
            1 => Some((i, BreakOpportunity::Allowed)),
            3 => Some((i, BreakOpportunity::Mandatory)),
            _ => None,
        })
        .chain((!text.is_empty()).then_some((text.len(), BreakOpportunity::Mandatory)))
}

/// `IDWriteTextAnalyzer::AnalyzeLineBreakpoints` over the UTF-16 `units`: one `DWRITE_LINE_BREAKPOINT` byte per unit
/// (breakConditionBefore:2, breakConditionAfter:2, isWhitespace:1, isSoftHyphen:1). `before` is the unit just before the
/// range and `after` the one the source reports after it, if any. Each unit starts may-not-break both ways; the pair walk
/// sets the condition between each two characters on both, the second unit of a surrogate pair keeping its own. The
/// first unit's condition before is neutral after a character (must-break after a mandatory break) and may-not with
/// none; the last unit's after is neutral before a character and otherwise can-break (must after a mandatory break).
pub fn line_breakpoints(units: &[u16], before: Option<u16>, after: Option<u16>) -> Vec<u8> {
    let n = units.len();
    if n == 0 {
        return Vec::new();
    }
    let mut cls = vec![40u8; n];
    let mut bp = vec![0u8; n];
    let mut text = String::with_capacity(n);
    let mut byte_unit = Vec::with_capacity(n + 1); // the unit each byte of `text` starts at
    let mut i = 0;
    for c in char::decode_utf16(units.iter().copied()) {
        let (cp, len) = match c {
            Ok(c) => (c as u32, c.len_utf16()),
            Err(e) => (e.unpaired_surrogate() as u32, 1),
        };
        let ch = char::from_u32(cp).unwrap_or('\u{FFFD}');
        cls[i] = class(cp);
        let init = (if cp == 0xAD { 0x2A } else { 0x0A }) | (if ch.is_whitespace() { 0x10 } else { 0 });
        bp[i..i + len].fill(init);
        byte_unit.extend(std::iter::repeat_n(i, ch.len_utf8()));
        text.push(ch);
        i += len;
    }
    byte_unit.push(n);
    // The Thai word breaker marks its word ends first.
    for b in super::super::thai::word_breaks(&text) {
        let u = byte_unit[b];
        bp[u - 1] = bp[u - 1] & 0xF3 | 4;
    }
    let first = cls[0];
    let start = before.map_or(2, |u| if action(class(u as u32), first) == 5 { 3 } else { 0 });
    bp[0] = bp[0] & 0xFC | start;
    let mut walk = Walk::new(first);
    for i in 1..n {
        if cls[i] >= 40 {
            continue;
        }
        let c = match walk.step(cls[i]) {
            0 => (bp[i - 1] >> 2) & 3,
            c => c,
        };
        bp[i - 1] = bp[i - 1] & 0xF3 | c << 2;
        bp[i] = bp[i] & 0xFC | c;
    }
    let end = match after {
        Some(u) => if action(walk.cls, class(u as u32)) == 5 { 3 } else { 0 },
        None => if action(walk.cls, 38) == 5 { 3 } else { 1 },
    };
    bp[n - 1] = bp[n - 1] & 0xF3 | end << 2;
    bp
}

#[cfg(test)]
mod tests {
    use super::*;

    /// VM layout clusters (Segoe UI, en-us): each string's canWrapLineAfter positions in UTF-16 units. A path breaks
    /// before each backslash after a letter but not after a digit, a URL after each slash before a letter, and a
    /// Hebrew letter keeps the hyphen after it unless a space parts them.
    #[test]
    fn breaks_match_native_clusters() {
        for (text, want) in [
            ("C:\\Users\\Public\\Documents\\Projects\\2026\\Quarterly", &[2, 8, 15, 25, 34][..]),
            ("/usr/local/bin/app --help", &[1, 5, 11, 15, 19, 21]),
            ("https://www.example.com/path/to/page?query=1&x=2#frag", &[8, 24, 29, 32, 37]),
            ("$1,234.56 USD", &[10]),
            ("hyphen-ated co-op/ed A/B and/or", &[7, 12, 15, 18, 21, 23, 25, 29]),
            ("soft\u{ad}hyphen non\u{2011}breaking", &[5, 12]),
            ("\u{5e9}\u{5dc}\u{5d5}\u{5dd}-\u{5e2}\u{5d5}\u{5dc}\u{5dd} \u{5d0}-\u{5d1} \u{5d0} -\u{5d1}", &[10, 14, 16, 17]),
            ("A+B=C x^2 a*b ~t `c` _u_ @m #h AT&T", &[1, 6, 10, 14, 17, 21, 25, 28, 31]),
            ("v1.2.3-beta+build.5", &[7, 11]),
        ] {
            let got: Vec<usize> = linebreaks(text, &[]).filter(|&(i, _)| i < text.len()).map(|(i, _)| text[..i].encode_utf16().count()).collect();
            assert_eq!(got, want, "{text}");
        }
    }

    /// `line_breakpoints` against native `AnalyzeLineBreakpoints` on the VM (`native_breakpoints.txt`, one hex line per
    /// case, `-` for none): `aXYa` and `aX Ya` for every pair of the classes' sample characters (lone surrogates, the
    /// mandatory breaks and a soft hyphen among them), then real strings whole and as every 1-, 2- and 3-unit range of
    /// them, whose neighbours come from the whole text (the one after as the last unit native reads there). Thai needs
    /// the dictionary: those cases run only with `D2D_THAI_DICT` naming a dwrite.dll.
    #[test]
    fn breakpoints_match_native_analyzer() {
        let thai = std::env::var_os("D2D_THAI_DICT").and_then(|p| std::fs::read(p).ok()).is_some_and(|d| super::super::super::thai::set_dictionary_from_dwrite(&d));
        let u = |s: &str| s.encode_utf16().collect::<Vec<u16>>();
        let reps: Vec<Vec<u16>> = [
            "\u{a7}", "a", "\u{2014}", "|", "\u{b4}", "\u{3041}", "}", "\u{301}", ")", "!", "\u{a0}", "\u{ac00}", "\u{ac01}", "\u{5d0}", "-",
            "\u{4e00}", "\u{2026}", ",", "\u{1100}", "\u{11a8}", "\u{1161}", "\u{3005}", "1", "(", "%", "$", "\"", "\u{1f1e6}", "\u{e01}", " ", "/",
            "\u{2060}", "\u{e000}", "\u{200b}", "\u{261d}", "\u{1f3fb}", "\u{200d}", "\u{c}", "\r", "\n", "\u{85}", "\u{fffc}", "\u{ad}", "\u{2011}",
            "\u{2028}", "\u{fffe}",
        ]
        .iter()
        .map(|s| u(s))
        .chain([vec![0xD800], vec![0xDC00]])
        .collect();
        let sp = 29; // the space's index above
        let mut cases: Vec<(Vec<u16>, usize, usize)> = Vec::new();
        for (x, rx) in reps.iter().enumerate() {
            for (y, ry) in reps.iter().enumerate() {
                let t = [&u("a")[..], rx, ry, &u("a")].concat();
                cases.push((t.clone(), 0, t.len()));
                if x != sp && y != sp {
                    let t = [&u("a")[..], rx, &u(" "), ry, &u("a")].concat();
                    cases.push((t.clone(), 0, t.len()));
                }
            }
        }
        let ws: String = [9u32, 10, 11, 12, 13, 0x20, 0x85, 0xa0, 0x1680].into_iter().chain(0x2000..0x200b).chain([0x2028, 0x2029, 0x202f, 0x205f, 0x3000]).filter_map(char::from_u32).collect();
        let mut real: Vec<Vec<u16>> = [
            "C:\\Users\\Public\\Documents\\Projects\\2026\\Quarterly Report (final).xlsx",
            "https://www.example.com/path/to/page?query=1&x=2#frag",
            "john.smith@example.com",
            "$1,234.56 USD 50% off \u{20ac}20.00",
            "(555) 012-3456",
            "hyphen-ated co-op/ed A/B and/or",
            "soft\u{ad}hyphen non\u{2011}breaking",
            "\u{65e5}\u{672c}\u{8a9e}\u{306e}\u{6587}\u{7ae0}\u{3067}\u{3059}\u{3002}\u{300c}\u{5f15}\u{7528}\u{300d}\u{ff08}\u{62ec}\u{5f27}\u{ff09}\u{3001}",
            "\u{3061}\u{3087}\u{3063}\u{3068}\u{30fc}\u{30d0}\u{30fc}",
            "\u{c548}\u{b155}\u{d558}\u{c138}\u{c694} \u{c138}\u{acc4}",
            "\u{5e9}\u{5dc}\u{5d5}\u{5dd}-\u{5e2}\u{5d5}\u{5dc}\u{5dd} \u{5d0}-\u{5d1} \u{5d0} -\u{5d1}",
            "\u{1f44d}\u{1f3fd} \u{1f468}\u{200d}\u{1f469}\u{200d}\u{1f467} \u{1f1fa}\u{1f1f8}",
            "\u{e2a}\u{e27}\u{e31}\u{e2a}\u{e14}\u{e35}\u{e04}\u{e23}\u{e31}\u{e1a} \u{e02}\u{e2d}\u{e1a}\u{e04}\u{e38}\u{e13}\u{e04}\u{e23}\u{e31}\u{e1a}",
            "line one\r\nline two\nthree\u{2028}four\u{2029}five\u{b}six",
        ]
        .iter()
        .map(|s| u(s))
        .collect();
        real.push(u(&(ws + "x")));
        real.push(u("v1.2.3-beta+build.5"));
        real.push(vec![0xD800, 0x78, 0xDC00, 0x79, 0xD800, 0xDC00, 0x7A]);
        real.extend(["", "a", " ", "\u{ad}", "\u{1f600}", "x\u{fffe}y\u{ffff}z"].iter().map(|s| u(s)));
        for t in &real {
            cases.push((t.clone(), 0, t.len()));
            for p in 0..=t.len() {
                for l in 1..=3 {
                    if p + l <= t.len() {
                        cases.push((t.clone(), p, l));
                    }
                }
            }
        }
        let want: Vec<&str> = include_str!("native_breakpoints.txt").lines().collect();
        assert_eq!(cases.len(), want.len());
        let mut bad = Vec::new();
        for ((t, p, l), w) in cases.iter().zip(&want) {
            if !thai && t.iter().any(|&c| (0xE01..=0xE5B).contains(&c)) {
                continue;
            }
            let before = (*p > 0).then(|| t[p - 1]);
            let after = (p + l < t.len()).then(|| t[t.len() - 1]);
            let got: String = line_breakpoints(&t[*p..p + l], before, after).iter().map(|b| format!("{b:02x}")).collect();
            let got = if got.is_empty() { "-".to_string() } else { got };
            if got != *w {
                bad.push(format!("{:x?} {p}+{l}: ours {got} native {w}", t));
            }
        }
        assert!(bad.is_empty(), "{} cases differ:\n{}", bad.len(), bad[..bad.len().min(20)].join("\n"));
    }
}
