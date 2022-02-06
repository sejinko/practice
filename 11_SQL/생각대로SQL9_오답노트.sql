-- 문제 2-1

with TB1 as
(
select 0 as chk1, 60 as chk2, 'short' as length_flag
union all
select 61 as chk1, 120 as chk2, 'middle' as length_flag
union all
select 121 as chk1, 9999 as chk2, 'long' as length_flag
)
select max(length)
from film f;


-- 문제 2-2

with TB1 as
(
select 0 as chk1, 60 as chk2, 'short' as length_flag
union all
select 61 as chk1, 120 as chk2, 'middle' as length_flag
union all
select 121 as chk1, 999 as chk2, 'long' as length_flag
)
select
	length_flag,
	count(distinct film_id) as cnt,
	count(film_id)
from
(
	select
		f.film_id,
		f.length,
		tb1.length_flag
	from
		film f
		left outer join TB1 on f.length between tb1.chk1 and tb1.chk2
) as db
group by 
	length_flag;
		