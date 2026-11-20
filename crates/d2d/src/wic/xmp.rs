//! The XMP (RDF/XML) packet WIC's metadata writer serialises, byte-for-byte.
//!
//! WIC's XMP writer is deterministic (not the full Adobe toolkit): a fixed `xpacket` id and
//! `rdf:about` uuid, one `<rdf:Description>` per namespace in first-use order, one nested element per
//! scalar property in set order, CRLF after the header and before the trailer, `end='w'` with no
//! padding. Properties use the namespace *prefix* from the query path (`/xmp/dc:format`), mapped to
//! its URI by the standard XMP namespace table. Verified byte-exact against native for the common
//! namespaces (JPEG APP1 / PNG iTXt). Arrays/structs and unlisted prefixes are out of scope.

/// XMP prefix → namespace URI, each verified byte-for-byte against native. A prefix absent here has
/// its property skipped (rather than risk guessing the wrong URI); add a row once verified.
const NS: &[(&str, &str)] = &[
    ("dc", "http://purl.org/dc/elements/1.1/"),
    ("xmp", "http://ns.adobe.com/xap/1.0/"),
    ("tiff", "http://ns.adobe.com/tiff/1.0/"),
    ("exif", "http://ns.adobe.com/exif/1.0/"),
    ("photoshop", "http://ns.adobe.com/photoshop/1.0/"),
    ("xmpMM", "http://ns.adobe.com/xap/1.0/mm/"),
];

/// XML text escaping, as native does for property values: `&`, `<`, `>`.
fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

/// Serialise scalar XMP properties `(prefix, local, value)` into the exact packet native writes, or
/// `None` if none have a known namespace prefix. Properties are grouped into one `rdf:Description`
/// per prefix in first-use order, preserving set order within each.
pub fn xmp_packet(props: &[(String, String, String)]) -> Option<Vec<u8>> {
    let mut order: Vec<&str> = Vec::new();
    for (pfx, _, _) in props {
        if NS.iter().any(|(p, _)| p == pfx) && !order.contains(&pfx.as_str()) {
            order.push(pfx);
        }
    }
    if order.is_empty() {
        return None;
    }
    let mut o = String::from("<?xpacket begin='\u{feff}' id='W5M0MpCehiHzreSzNTczkc9d'?>\r\n");
    o.push_str("<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">");
    for pfx in &order {
        let uri = NS.iter().find(|(p, _)| p == pfx).unwrap().1;
        o.push_str(&format!(
            "<rdf:Description rdf:about=\"uuid:faf5bdd5-ba3d-11da-ad31-d33d75182f1b\" xmlns:{pfx}=\"{uri}\">"
        ));
        for (p, local, val) in props {
            if p == pfx {
                o.push_str(&format!("<{pfx}:{local}>{}</{pfx}:{local}>", esc(val)));
            }
        }
        o.push_str("</rdf:Description>");
    }
    o.push_str("</rdf:RDF></x:xmpmeta>\r\n<?xpacket end='w'?>");
    Some(o.into_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grouping_order_and_escape() {
        // Two namespaces, interleaved sets, grouped in first-use order; `<`/`&`/`>` escaped.
        let p = |a: &str, b: &str, c: &str| (a.into(), b.into(), c.into());
        let out = xmp_packet(&[p("xmp", "CreatorTool", "Tool"), p("dc", "title", "a<b>&c"), p("xmp", "Label", "L")]).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("<xmp:CreatorTool>Tool</xmp:CreatorTool><xmp:Label>L</xmp:Label>"));
        assert!(s.contains("<dc:title>a&lt;b&gt;&amp;c</dc:title>"));
        assert!(s.ends_with("</rdf:RDF></x:xmpmeta>\r\n<?xpacket end='w'?>"));
        assert!(xmp_packet(&[p("bogus", "X", "Y")]).is_none());
    }
}
