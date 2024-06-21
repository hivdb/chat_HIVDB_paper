SELECT
    COUNT(1)
FROM
    tblRefLink
WHERE
    RefID IN (
        SELECT RefID
        FROM tblReferences
        WHERE MedlineID = {pubmed_id}
    )
;
