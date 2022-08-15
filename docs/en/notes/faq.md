# Frequently Asked Questions

We list some common troubles faced by many users and their corresponding solutions here. Feel free to enrich the list if you find any frequent issues and have ways to help others to solve them. If the contents here do not cover your issue, please create an issue using the [provided templates](https://github.com/open-mmlab/mmflow/blob/master/.github/ISSUE_TEMPLATE/error-report.md/) and make sure you fill in all required
information in the template.

## Installation

The compatible MMFlow and MMCV versions are as below. Please install the correct version of MMCV to avoid installation issues.

<table>
    <thead>
        <tr>
            <td>MMFlow version</td>
            <td>MMCV version</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th>master</th>
            <th>mmcv-full>=1.3.15, <1.7.0</th>
        </tr>
    </tbody>
    <tbody>
        <tr>
            <th>0.5.1</th>
            <th>mmcv-full>=1.3.15, <1.7.0</th>
        </tr>
    </tbody>
    <tbody>
        <tr>
            <th>0.5.0</th>
            <th>mmcv-full>=1.3.15, <=1.6.0</th>
        </tr>
    </tbody>
    <tbody>
        <tr>
            <th>0.4.2</th>
            <th>mmcv-full>=1.3.15, <=1.6.0</th>
        </tr>
    </tbody>
    <tbody>
        <tr>
            <th>0.4.1</th>
            <th>mmcv-full>=1.3.15, <=1.6.0</th>
        </tr>
    </tbody>
    <tbody>
        <tr>
            <th>0.4.0</th>
            <th>mmcv-full>=1.3.15, <=1.5.0</th>
        </tr>
    </tbody>
    <tbody>
        <tr>
            <th>0.3.0</th>
            <th>mmcv-full>=1.3.15, <=1.5.0</th>
        </tr>
    </tbody>
    <tbody>
        <tr>
            <th>0.2.0</th>
            <th>mmcv-full>=1.3.15, <=1.5.0</th>
        </tr>
    </tbody>
</table>

You need to run `pip uninstall mmcv` first if you have mmcv installed. If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.
