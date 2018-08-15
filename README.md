# Code for generating my personal website.

1. Install dependencies

```
pip install pelican markdown
```

2. To update and regenerate the website, update `content/pages` and run
```
>> pelican content
>> cd output
>> python -m pelican.server
```
You can preview the website in `http://localhost:8000`.

3. To publish, run
```
>> make github
```
