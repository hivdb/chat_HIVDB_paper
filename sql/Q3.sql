SELECT
    DISTINCT AccessionID
FROM
    tblSequences
WHERE
    IsolateID IN (
        SELECT IsolateID
        FROM tblRefLink
        WHERE RefID IN (
            SELECT RefID
            FROM tblReferences
            WHERE MedlineID = {pubmed_id}
        )
    )
    AND
    AccessionID IS NOT NULL
ORDER BY
    AccessionID
;
