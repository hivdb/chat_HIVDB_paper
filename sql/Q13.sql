SELECT
    DISTINCT Source
FROM
    tblClinIsolates
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
;